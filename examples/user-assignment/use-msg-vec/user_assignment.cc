#include "user_assignment.h"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/network-module.h"
#include "ns3/okumura-hata-propagation-loss-model.h"
#include "ns3/point-to-point-module.h"
#include "ns3/ai-module.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("UserAssignmentTemplate");

// ===========================
// 1. CONFIGURATION STRUCTS
// ===========================
struct SimConfig
{
    uint32_t numEnbs = 2;
    uint32_t numUes  = 20;
    double   simTime = 2000.0;

    double interSiteDistanceX = 1000.0;
    double interSiteDistanceY = 1000.0;

    // eNB columns per row for GridPositionAllocator. 0 = auto (all eNBs in
    // a single row, i.e. the original layout). Set to e.g. 2 for a 2x2
    // grid with numEnbs=4.
    uint32_t enbGridCols = 0;

    double assignmentInterval = 1.0;

    // Minimum real time a UE must stay on a cell before another handover can
    // be triggered for it. Real LTE handover algorithms enforce this kind of
    // hysteresis/time-to-trigger to prevent ping-pong; without it, an
    // exploring (near-random) RL policy can re-target a UE before its
    // previous X2 handover's spectrum-listener bookkeeping has settled.
    double handoverCooldown = 2.0;

    bool        enableNetAnim = true;
    std::string animFile      = "user-assignment.xml";
    std::string featureCsv    = "rl-feature-log.csv";
    std::string mobilityCsv   = "ue-mobility-log.csv";
};

struct FeatureRow
{
    double   timeSec    = 0.0;
    uint32_t ueId       = 0;

    double ux = 0.0;
    double uy = 0.0;
    double vx = 0.0;
    double vy = 0.0;

    int32_t servingCell = -1;

    std::vector<double> sinrDb;       // per-cell SINR  [dB]
    std::vector<double> cellLoad;     // per-cell load  [0,1]
    std::vector<double> cellCapacity; // per-cell static allocated capacity [Mbps] (constant)

    double servingSinrDb = -200.0;    // real PHY SINR of the current serving cell only [dB]
    std::vector<double> rsrpDbm;      // per-cell RSRP [dBm], incl. non-serving cells

    int32_t targetBestCell = -1;      // oracle: highest-SINR cell
};

// ===========================
// 2. RUNTIME STATE
// ===========================
constexpr uint32_t MAX_UES = 100;

struct RuntimeState
{
    SimConfig cfg;

    std::vector<int32_t>  servingCell;
    std::vector<uint32_t> cellUserCounts;
    std::vector<double>   cellLoad;
    std::vector<double>   cellCapacity; // static allocated Mbps per cell (constant, set once)
    std::vector<double>   lastHandoverTimeSec; // per-UE, for cooldown gating

    uint32_t step   = 0;
    double   lastReward               = 0.0;
    double   lastTotalThroughputMbps  = 0.0;
    double   lastLoadStd              = 0.0;
    uint32_t lastHandovers            = 0;
    uint32_t lastOutages              = 0;

    std::ofstream csv;
    std::ofstream mobilityCsv;
};

// ===========================
// 3. NS3-AI INTERFACE GLOBAL
// ===========================
static Ns3AiMsgInterfaceImpl<UserAssignmentEnv, UserAssignmentAct>* g_msgInterface = nullptr;

// ===========================
// 4. LTE GLOBAL STATE
// ===========================
static Ptr<LteHelper>             g_lteHelper  = nullptr;
static Ptr<PointToPointEpcHelper> g_epcHelper  = nullptr;
static NetDeviceContainer         g_enbDevs;
static NetDeviceContainer         g_ueDevs;

// Per-UE LTE measurements updated by trace callbacks
struct UeLteMeasurement
{
    double   sinrLinear    = 0.0;   // serving-cell SINR, linear (from ReportCurrentCellRsrpSinr)
    double   rsrpWatts     = 0.0;   // serving-cell RSRP, W      (from ReportCurrentCellRsrpSinr)
    uint16_t servingCellId = 0;     // LTE cell-ID (1-indexed)
    std::vector<double> rsrpPerCellDbm; // indexed 0..numEnbs-1, dBm (from ReportUeMeasurements)
};
static std::vector<UeLteMeasurement> g_ueLteMeas;

// Per-cell DL bytes scheduled since last interval reset (from DlScheduling callback)
static std::vector<uint64_t> g_cellBytesScheduled;

// Per-UE PacketSink + baseline for delta throughput
static std::vector<Ptr<PacketSink>> g_packetSinks;
static std::vector<uint64_t>        g_ueBytesAtLastInterval;

// LTE 20-MHz thermal noise floor (-174 dBm/Hz + 10·log10(20e6) + 7 dB NF = -94 dBm)
constexpr double LTE_NOISE_FLOOR_DBM  = -94.0;
// Approximate 20-MHz LTE DL peak (used for load fraction denominator)
constexpr double LTE_PEAK_DL_MBPS    = 150.0;

// ===========================
// 5. LTE TRACE CALLBACKS
// ===========================

// Fires ~every 1 ms per UE from its serving cell.
// rsrpW     = average RS received power [W] (linear)
// sinrLinear = average SINR over RBs    (linear)
static void
RsrpSinrCallback(uint32_t ueIdx,
                 uint16_t cellId,
                 uint16_t /*rnti*/,
                 double   rsrpW,
                 double   sinrLinear,
                 uint8_t  /*ccId*/)
{
    if (ueIdx >= g_ueLteMeas.size())
        return;
    g_ueLteMeas[ueIdx].sinrLinear    = sinrLinear;
    g_ueLteMeas[ueIdx].rsrpWatts     = rsrpW;
    g_ueLteMeas[ueIdx].servingCellId = cellId;
}

// Fires ~every 200 ms per UE, once per measured cell.
// rsrpDbm = RSRP in dBm (already converted inside LteUePhy)
static void
UeMeasCallback(uint32_t ueIdx,
               uint16_t /*rnti*/,
               uint16_t cellId,
               double   rsrpDbm,
               double   /*rsrqDb*/,
               bool     /*isServing*/,
               uint8_t  /*ccId*/)
{
    if (ueIdx >= g_ueLteMeas.size())
        return;
    const uint32_t cellIdx = static_cast<uint32_t>(cellId) - 1; // 1-indexed → 0-indexed
    if (cellIdx < g_ueLteMeas[ueIdx].rsrpPerCellDbm.size())
        g_ueLteMeas[ueIdx].rsrpPerCellDbm[cellIdx] = rsrpDbm;
}

// Fires every subframe for each UE scheduled by the PF MAC.
// Accumulates DL transport-block bytes per cell for load/capacity estimation.
static void
DlSchedulingCallback(uint32_t cellIdx, DlSchedulingCallbackInfo info)
{
    if (cellIdx >= g_cellBytesScheduled.size())
        return;
    g_cellBytesScheduled[cellIdx] += info.sizeTb1;
    if (info.sizeTb2 > 0)
        g_cellBytesScheduled[cellIdx] += info.sizeTb2;
}

// ===========================
// 6. LTE DATA QUERY FUNCTIONS  (replace all proxy functions)
// ===========================

// Compute per-cell SINR [dB] for a given UE from real LTE measurements.
// Non-serving cells: SINR = RSRP[b] / (sum_other RSRP + noise)  using ReportUeMeasurements.
// Serving cell     : override with real PHY SINR from ReportCurrentCellRsrpSinr.
static std::vector<double>
GetSinrPerCell(uint32_t ueIdx, uint32_t numEnbs)
{
    std::vector<double> sinrDb(numEnbs, -100.0);

    // Convert per-cell RSRP from dBm → linear mW
    std::vector<double> rsrpMw(numEnbs);
    for (uint32_t b = 0; b < numEnbs; ++b)
        rsrpMw[b] = std::pow(10.0, g_ueLteMeas[ueIdx].rsrpPerCellDbm[b] / 10.0);

    const double noiseMw = std::pow(10.0, LTE_NOISE_FLOOR_DBM / 10.0);

    for (uint32_t b = 0; b < numEnbs; ++b)
    {
        double interference = noiseMw;
        for (uint32_t i = 0; i < numEnbs; ++i)
            if (i != b)
                interference += rsrpMw[i];
        sinrDb[b] = 10.0 * std::log10(rsrpMw[b] / interference);
    }

    // Override serving-cell entry with real PHY SINR (includes all real interference effects)
    const uint16_t scId = g_ueLteMeas[ueIdx].servingCellId;
    if (scId > 0 && static_cast<uint32_t>(scId - 1) < numEnbs)
    {
        const double lin = g_ueLteMeas[ueIdx].sinrLinear;
        if (lin > 0.0)
            sinrDb[scId - 1] = 10.0 * std::log10(lin);
    }

    return sinrDb;
}

// Snapshot DL scheduling bytes → cell load fraction [0,1].
// cellCapacity is the cell's static allocated capacity (constant, set once at
// startup) and is only read here as the load denominator, never modified.
// Resets g_cellBytesScheduled — call exactly once per interval.
static void
SnapshotCellStats(uint32_t                   numEnbs,
                  double                     intervalSeconds,
                  std::vector<double>&       cellLoad,
                  const std::vector<double>& cellCapacity)
{
    cellLoad.resize(numEnbs);

    for (uint32_t b = 0; b < numEnbs; ++b)
    {
        const double mbps =
            static_cast<double>(g_cellBytesScheduled[b]) * 8.0 / 1e6 / intervalSeconds;
        cellLoad[b]              = std::min(1.0, mbps / cellCapacity[b]);
        g_cellBytesScheduled[b] = 0; // reset for next interval
    }
}

// Real DL throughput [Mbps] delivered to a UE, measured from PacketSink byte delta.
static double
GetUeThroughputMbps(uint32_t ueIdx, double intervalSeconds)
{
    if (ueIdx >= g_packetSinks.size() || !g_packetSinks[ueIdx])
        return 0.0;
    const uint64_t totalRx = g_packetSinks[ueIdx]->GetTotalRx();
    const uint64_t delta   = totalRx - g_ueBytesAtLastInterval[ueIdx];
    g_ueBytesAtLastInterval[ueIdx] = totalRx;
    return static_cast<double>(delta) * 8.0 / 1e6 / intervalSeconds;
}

// Oracle best-cell: cell with highest SINR for this UE.
static int32_t
ChooseBestCellBySinr(const std::vector<double>& sinrDb)
{
    int32_t best = 0;
    for (uint32_t b = 1; b < sinrDb.size(); ++b)
        if (sinrDb[b] > sinrDb[best])
            best = static_cast<int32_t>(b);
    return best;
}

// ===========================
// 7. STATISTICS HELPERS
// ===========================
static double
Mean(const std::vector<double>& v)
{
    if (v.empty())
        return 0.0;
    double s = 0.0;
    for (double x : v)
        s += x;
    return s / static_cast<double>(v.size());
}

static double
StdDev(const std::vector<double>& v)
{
    if (v.empty())
        return 0.0;
    const double m = Mean(v);
    double acc = 0.0;
    for (double x : v)
    {
        const double d = x - m;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(v.size()));
}

// ===========================
// 8. COMMAND-LINE PARSING
// ===========================
static void
ParseArgs(int argc, char* argv[], SimConfig& cfg)
{
    CommandLine cmd(__FILE__);
    cmd.AddValue("numEnbs",            "Number of eNBs",                    cfg.numEnbs);
    cmd.AddValue("numUes",             "Number of UEs",                     cfg.numUes);
    cmd.AddValue("simTime",            "Simulation time (s)",               cfg.simTime);
    cmd.AddValue("interSiteDistanceX", "Horizontal inter-site distance (m)",cfg.interSiteDistanceX);
    cmd.AddValue("interSiteDistanceY", "Vertical inter-site distance (m)",  cfg.interSiteDistanceY);
    cmd.AddValue("enbGridCols",        "eNB columns per row (0=single row)",cfg.enbGridCols);
    cmd.AddValue("assignmentInterval", "RL reassignment interval (s)",      cfg.assignmentInterval);
    cmd.AddValue("animFile",           "NetAnim XML output",                cfg.animFile);
    cmd.AddValue("featureCsv",         "Feature/training CSV",              cfg.featureCsv);
    cmd.AddValue("mobilityCsv",        "UE mobility CSV",                   cfg.mobilityCsv);
    cmd.Parse(argc, argv);
}

// ===========================
// 9. NODE CREATION
// ===========================
static void
CreateNodes(const SimConfig& cfg, NodeContainer& enbNodes, NodeContainer& ueNodes)
{
    enbNodes.Create(cfg.numEnbs);
    ueNodes.Create(cfg.numUes);
}

// ===========================
// 10. ENB MOBILITY
// ===========================
// eNB grid origin — also anchors the UE-mobility coverage rectangle below,
// since both must agree on where the eNBs actually sit.
constexpr double kEnbGridMinX = 500.0;
constexpr double kEnbGridMinY = 500.0;

static void
InstallEnbMobility(const SimConfig& cfg, NodeContainer& enbNodes)
{
    const uint32_t gridCols = (cfg.enbGridCols > 0) ? cfg.enbGridCols : cfg.numEnbs;

    MobilityHelper mob;
    mob.SetPositionAllocator("ns3::GridPositionAllocator",
                             "MinX",      DoubleValue(kEnbGridMinX),
                             "MinY",      DoubleValue(kEnbGridMinY),
                             "DeltaX",    DoubleValue(cfg.interSiteDistanceX),
                             "DeltaY",    DoubleValue(cfg.interSiteDistanceY),
                             "GridWidth", UintegerValue(gridCols),
                             "LayoutType",StringValue("RowFirst"));
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mob.Install(enbNodes);

    // Okumura-Hata requires base-station height > 0.  30 m = typical macro cell.
    for (uint32_t i = 0; i < enbNodes.GetN(); ++i)
    {
        Ptr<MobilityModel> m = enbNodes.Get(i)->GetObject<MobilityModel>();
        Vector p = m->GetPosition();
        p.z = 30.0;
        m->SetPosition(p);
    }
}

// ===========================
// 11. UE MOBILITY
// ===========================
static void
InstallUeMobility(const SimConfig& cfg, NodeContainer& ueNodes)
{
    MobilityHelper mob;

    // UE movement area sized to the eNBs' coverage footprint: a half-ISD
    // margin around the eNB grid (the natural cell-edge distance between
    // adjacent eNBs), so UEs stay within reasonable SINR range instead of
    // wandering into locations no cell covers well.
    const uint32_t gridCols = (cfg.enbGridCols > 0) ? cfg.enbGridCols : cfg.numEnbs;
    const uint32_t gridRows = (cfg.numEnbs + gridCols - 1) / gridCols; // ceil

    const double marginX  = cfg.interSiteDistanceX / 2.0;
    const double marginY  = (cfg.interSiteDistanceY > 0.0 ? cfg.interSiteDistanceY : cfg.interSiteDistanceX) / 2.0;
    const double enbMaxX  = kEnbGridMinX + static_cast<double>(gridCols - 1) * cfg.interSiteDistanceX;
    const double enbMaxY  = kEnbGridMinY + static_cast<double>(gridRows - 1) * cfg.interSiteDistanceY;
    const double areaMinX = kEnbGridMinX - marginX;
    const double areaMaxX = enbMaxX      + marginX;
    const double areaMinY = kEnbGridMinY - marginY;
    const double areaMaxY = enbMaxY      + marginY;

    Ptr<RandomRectanglePositionAllocator> posAlloc =
        CreateObject<RandomRectanglePositionAllocator>();
    posAlloc->SetAttribute("X", StringValue(
        "ns3::UniformRandomVariable[Min=" + std::to_string(areaMinX) +
        "|Max=" + std::to_string(areaMaxX) + "]"));
    posAlloc->SetAttribute("Y", StringValue(
        "ns3::UniformRandomVariable[Min=" + std::to_string(areaMinY) +
        "|Max=" + std::to_string(areaMaxY) + "]"));
    mob.SetPositionAllocator(posAlloc);

    mob.SetMobilityModel("ns3::RandomDirection2dMobilityModel",
                         "Bounds", RectangleValue(Rectangle(areaMinX, areaMaxX,
                                                            areaMinY, areaMaxY)),
                         "Speed",  StringValue("ns3::ConstantRandomVariable[Constant=5.0]"),
                         "Pause",  StringValue("ns3::ConstantRandomVariable[Constant=1.0]"));
    mob.Install(ueNodes);

    // Okumura-Hata requires mobile height > 0.  1.5 m = pedestrian.
    // RandomDirection2d only moves in x/y, so z=1.5 is preserved throughout.
    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        Ptr<MobilityModel> m = ueNodes.Get(u)->GetObject<MobilityModel>();
        Vector p = m->GetPosition();
        p.z = 1.5;
        m->SetPosition(p);
    }
}

// ===========================
// 12. LTE STACK + EPC SETUP
// ===========================
static void
SetupLteStack(const SimConfig&        cfg,
              NodeContainer&          enbNodes,
              NodeContainer&          ueNodes,
              Ptr<Node>&              remoteHostOut,
              Ipv4InterfaceContainer& ueIpIfaceOut)
{
    // --- LteHelper: scheduler / pathloss / antenna / bandwidth ---
    g_lteHelper = CreateObject<LteHelper>();

    // Proportional-Fair scheduler: uses real CQI feedback from LteAmc
    g_lteHelper->SetSchedulerType("ns3::PfFfMacScheduler");

    // Okumura-Hata medium-city urban pathloss.
    // ns-3's OkumuraHataPropagationLossModel only reads the "Environment"
    // attribute (Urban/SubUrban/OpenAreas) in its classic-Hata branch, which
    // is gated on Frequency <= 1.5GHz -- above that it switches to the
    // COST-231 formula, which ignores Environment entirely and only
    // consults CitySize. The default EARFCN (Band 1, ~2.11GHz) silently put
    // us in that COST-231 branch, so "Environment" was being set but never
    // read. Band 8 (~900MHz) keeps us under the 1.5GHz cutover so Environment
    // actually takes effect.
    g_lteHelper->SetPathlossModelType(
        TypeId::LookupByName("ns3::OkumuraHataPropagationLossModel"));
    g_lteHelper->SetPathlossModelAttribute("Environment", EnumValue(SubUrbanEnvironment));
    g_lteHelper->SetPathlossModelAttribute("CitySize",    EnumValue(MediumCity));

    // Isotropic (omni-directional) antenna on both eNB and UE sides
    g_lteHelper->SetEnbAntennaModelType("ns3::IsotropicAntennaModel");
    g_lteHelper->SetUeAntennaModelType ("ns3::IsotropicAntennaModel");

    // 20 MHz = 100 resource blocks (DL and UL)
    g_lteHelper->SetEnbDeviceAttribute("DlBandwidth", UintegerValue(100));
    g_lteHelper->SetEnbDeviceAttribute("UlBandwidth", UintegerValue(100));

    // Band 8 (900MHz E-GSM): DL EARFCN 3450 -> 925MHz, UL EARFCN 21450 -> 880MHz.
    // Keeps Frequency <= 1.5GHz so the pathloss model's Environment attribute
    // (set above) is actually consulted -- see comment above.
    // NB: ns-3's internal EARFCN table (lte-spectrum-value-helper.cc) uses a
    // flat "DL offset + 18000" rule for every band's UL offset, which does
    // NOT match real 3GPP band 8 (whose actual UL offset is 20000, i.e.
    // EARFCN 23450) -- using the real-world value here throws
    // NS_ASSERT_MSG(fc != 0, "invalid EARFCN=...") because it falls in a gap
    // ns-3's table doesn't cover. Must match ns-3's own table, not the spec.
    g_lteHelper->SetEnbDeviceAttribute("DlEarfcn", UintegerValue(3450));
    g_lteHelper->SetEnbDeviceAttribute("UlEarfcn", UintegerValue(21450));

    // NoOpHandoverAlgorithm is already the default in ns-3.46 LteHelper, meaning
    // the stack never initiates its own handovers.  The RL agent issues all
    // handovers explicitly via g_lteHelper->HandoverRequest().

    // --- EPC (Evolved Packet Core) enables real IP downlink data ---
    g_epcHelper = CreateObject<PointToPointEpcHelper>();
    g_lteHelper->SetEpcHelper(g_epcHelper);

    // --- Install eNB and UE LTE devices ---
    g_enbDevs = g_lteHelper->InstallEnbDevice(enbNodes);
    g_ueDevs  = g_lteHelper->InstallUeDevice(ueNodes);

    // Set macro-cell Tx power per eNB (43–45 dBm)
    for (uint32_t b = 0; b < enbNodes.GetN(); ++b)
    {
        const double txPower = 43.0 + static_cast<double>(b % 3);
        Ptr<LteEnbNetDevice> dev = g_enbDevs.Get(b)->GetObject<LteEnbNetDevice>();
        dev->GetPhy()->SetTxPower(txPower);
    }

    // --- Remote host: UDP traffic source for downlink ---
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);

    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    // Connect remote host to EPC PGW via high-speed P2P (no bandwidth bottleneck)
    Ptr<Node> pgw = g_epcHelper->GetPgwNode();
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute ("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetDeviceAttribute ("Mtu",      UintegerValue(1500));
    p2ph.SetChannelAttribute("Delay",    TimeValue(MilliSeconds(0)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIfaces = ipv4h.Assign(internetDevices);

    // Static route: remote host → UE subnet (7.0.0.0/8 is EPC default UE pool)
    Ipv4StaticRoutingHelper routingHelper;
    Ptr<Ipv4StaticRouting> remoteHostRouting =
        routingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"),
                                         Ipv4Mask("255.0.0.0"), 1);

    // --- Internet stack + IP addresses on UEs ---
    internet.Install(ueNodes);
    ueIpIfaceOut = g_epcHelper->AssignUeIpv4Address(NetDeviceContainer(g_ueDevs));

    // Default gateway for each UE (EPC SGW)
    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        Ptr<Ipv4StaticRouting> ueRouting =
            routingHelper.GetStaticRouting(ueNodes.Get(u)->GetObject<Ipv4>());
        ueRouting->SetDefaultRoute(g_epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    remoteHostOut = remoteHost;
}

// ===========================
// 13. SATURATING UDP DOWNLINK TRAFFIC
// ===========================
// Rate per UE: 1400 B × 5000 pkt/s ≈ 56 Mbps.
// LTE 20 MHz peak per cell ≈ 150 Mbps shared across ~10 UEs → ~15 Mbps/UE max.
// Offered load >> capacity, so the LTE scheduler is always the bottleneck.
// Delivered bytes at PacketSink = real LTE throughput driven by CQI + PF scheduling.
static void
SetupUdpTraffic(const SimConfig&              cfg,
                NodeContainer&                ueNodes,
                const Ipv4InterfaceContainer& ueIpIface,
                Ptr<Node>                     remoteHost)
{
    const uint16_t basePort = 1100;

    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        const uint16_t dlPort = basePort + u + 1;

        // PacketSink on each UE receives downlink UDP
        PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
                                    InetSocketAddress(Ipv4Address::GetAny(), dlPort));
        ApplicationContainer sinkApp = sinkHelper.Install(ueNodes.Get(u));
        sinkApp.Start(Seconds(0.1));
        sinkApp.Stop(Seconds(cfg.simTime));
        g_packetSinks.push_back(sinkApp.Get(0)->GetObject<PacketSink>());

        // UdpClient on remote host → this UE
        UdpClientHelper clientHelper(ueIpIface.GetAddress(u), dlPort);
        clientHelper.SetAttribute("Interval",   TimeValue(MicroSeconds(200))); // 5000 pkt/s
        clientHelper.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
        clientHelper.SetAttribute("PacketSize", UintegerValue(1400));

        ApplicationContainer clientApp = clientHelper.Install(remoteHost);
        clientApp.Start(Seconds(0.1));
        clientApp.Stop(Seconds(cfg.simTime));
    }
}

// ===========================
// 14. LTE TRACE CALLBACK CONNECTIONS
// ===========================
static void
ConnectLteCallbacks(const SimConfig& cfg,
                    NodeContainer&   ueNodes,
                    NodeContainer&   enbNodes)
{
    // Per-UE: serving-cell RSRP/SINR trace (~1 ms period) and
    // neighbor-cell measurement trace (~200 ms period)
    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        Ptr<LteUePhy> phy =
            g_ueDevs.Get(u)->GetObject<LteUeNetDevice>()->GetPhy();

        phy->TraceConnectWithoutContext(
            "ReportCurrentCellRsrpSinr",
            MakeBoundCallback(&RsrpSinrCallback, u));

        phy->TraceConnectWithoutContext(
            "ReportUeMeasurements",
            MakeBoundCallback(&UeMeasCallback, u));
    }

    // Per-eNB: DL scheduling trace (fires each subframe per scheduled UE)
    for (uint32_t b = 0; b < enbNodes.GetN(); ++b)
    {
        Ptr<LteEnbMac> mac =
            g_enbDevs.Get(b)->GetObject<LteEnbNetDevice>()->GetMac();

        mac->TraceConnectWithoutContext(
            "DlScheduling",
            MakeBoundCallback(&DlSchedulingCallback, b));
    }
}

// ===========================
// 15. INITIAL NEAREST-CELL ASSIGNMENT
// ===========================
static std::vector<int32_t>
InitialNearestCellAssignment(const NodeContainer& enbNodes,
                              const NodeContainer& ueNodes)
{
    std::vector<int32_t> serving(ueNodes.GetN(), -1);

    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        const Vector uePos =
            ueNodes.Get(u)->GetObject<MobilityModel>()->GetPosition();

        double  bestDist = std::numeric_limits<double>::infinity();
        int32_t bestCell = -1;

        for (uint32_t b = 0; b < enbNodes.GetN(); ++b)
        {
            const Vector ep =
                enbNodes.Get(b)->GetObject<MobilityModel>()->GetPosition();
            const double dx = uePos.x - ep.x;
            const double dy = uePos.y - ep.y;
            const double d  = std::sqrt(dx * dx + dy * dy);
            if (d < bestDist)
            {
                bestDist = d;
                bestCell = static_cast<int32_t>(b);
            }
        }
        serving[u] = bestCell;
    }
    return serving;
}

static std::vector<uint32_t>
ComputeCellUserCounts(const SimConfig& cfg, const std::vector<int32_t>& servingCell)
{
    std::vector<uint32_t> counts(cfg.numEnbs, 0);
    for (int32_t cell : servingCell)
        if (cell >= 0 && static_cast<uint32_t>(cell) < cfg.numEnbs)
            counts[static_cast<uint32_t>(cell)]++;
    return counts;
}

// ===========================
// 16. REWARD METRICS
// ===========================
struct RewardMetrics
{
    double   totalThroughputMbps = 0.0;
    double   loadStd             = 0.0;
    uint32_t handovers           = 0;
    uint32_t outages             = 0;
    double   reward              = 0.0;
};

static RewardMetrics
ComputeRewardMetrics(RuntimeState& state, const NodeContainer& ueNodes)
{
    RewardMetrics m;
    m.handovers = state.lastHandovers;
    m.loadStd   = StdDev(state.cellLoad);

    constexpr double outageSinrThreshDb  = -5.0;
    constexpr double outageThroughputMbps = 0.5;

    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        // Real delivered bytes from PacketSink delta
        const double ueThrpt = GetUeThroughputMbps(u, state.cfg.assignmentInterval);
        m.totalThroughputMbps += ueThrpt;

        // Outage: real serving-cell SINR below threshold OR throughput too low
        const double lin    = g_ueLteMeas[u].sinrLinear;
        const double sinrDb = (lin > 0.0) ? 10.0 * std::log10(lin) : -200.0;
        if (sinrDb < outageSinrThreshDb || ueThrpt < outageThroughputMbps)
            m.outages++;
    }

    m.reward = m.totalThroughputMbps
               - 5.0  * m.loadStd
               - 2.0  * static_cast<double>(m.handovers)
               - 10.0 * static_cast<double>(m.outages);

    return m;
}

// ===========================
// 17. FEATURE EXTRACTION
// ===========================
static FeatureRow
BuildFeatureRow(const RuntimeState& state,
                const NodeContainer& ueNodes,
                uint32_t ueId)
{
    FeatureRow row;
    row.timeSec     = Simulator::Now().GetSeconds();
    row.ueId        = ueId;
    row.servingCell = state.servingCell.at(ueId);

    Ptr<MobilityModel> mob = ueNodes.Get(ueId)->GetObject<MobilityModel>();
    const Vector p = mob->GetPosition();
    const Vector v = mob->GetVelocity();
    row.ux = p.x;  row.uy = p.y;
    row.vx = v.x;  row.vy = v.y;

    // Real per-cell SINR from LTE PHY measurements
    row.sinrDb = GetSinrPerCell(ueId, state.cfg.numEnbs);

    // Cell load and capacity already snapshotted this interval
    row.cellLoad     = state.cellLoad;
    row.cellCapacity = state.cellCapacity;

    // Real PHY SINR of the serving cell only (same value GetSinrPerCell uses
    // to override sinrDb[servingCell]; kept as its own column since SINR to
    // non-serving cells is only an RSRP-based approximation, not a real measurement).
    const double servingLin = g_ueLteMeas[ueId].sinrLinear;
    row.servingSinrDb = (servingLin > 0.0) ? 10.0 * std::log10(servingLin) : -200.0;

    // RSRP to every cell, including non-serving ones (from ReportUeMeasurements)
    row.rsrpDbm = g_ueLteMeas[ueId].rsrpPerCellDbm;

    // Oracle: best cell by highest real SINR
    row.targetBestCell = ChooseBestCellBySinr(row.sinrDb);

    return row;
}

// ===========================
// 18. CSV OUTPUT
// ===========================
static void
WriteCsvHeader(RuntimeState& state)
{
    state.csv.open(state.cfg.featureCsv, std::ios::out);
    if (!state.csv.is_open())
    {
        NS_LOG_UNCOND("Failed to open feature CSV: " << state.cfg.featureCsv);
        return;
    }
    state.csv << "step,time,ueId,servingCell,reward,totalThroughputMbps,"
                 "loadStd,handovers,outages";
    for (uint32_t b = 0; b < state.cfg.numEnbs; ++b)
        state.csv << ",load_bs" << b;
    for (uint32_t b = 0; b < state.cfg.numEnbs; ++b)
        state.csv << ",capacity_bs" << b;
    state.csv << ",servingSinrDb";
    for (uint32_t b = 0; b < state.cfg.numEnbs; ++b)
        state.csv << ",rsrp_bs" << b;
    state.csv << ",targetBestCell\n";
}

static void
WriteMobilityCsvHeader(RuntimeState& state, const NodeContainer& enbNodes)
{
    state.mobilityCsv.open(state.cfg.mobilityCsv, std::ios::out);
    if (!state.mobilityCsv.is_open())
    {
        NS_LOG_UNCOND("Failed to open mobility CSV: " << state.cfg.mobilityCsv);
        return;
    }
    state.mobilityCsv << "step,time,ueId,x,y,vx,vy,servingCell";
    for (uint32_t b = 0; b < enbNodes.GetN(); ++b)
        state.mobilityCsv << ",dist_bs" << b;
    state.mobilityCsv << "\n";
}

static void
LogFeatureRow(RuntimeState& state, const FeatureRow& row)
{
    if (!state.csv.is_open())
        return;

    state.csv << std::fixed << std::setprecision(3)
              << state.step << ","
              << row.timeSec << ","
              << row.ueId << ","
              << row.servingCell << ","
              << state.lastReward << ","
              << state.lastTotalThroughputMbps << ","
              << state.lastLoadStd << ","
              << state.lastHandovers << ","
              << state.lastOutages;

    for (double v : row.cellLoad)    state.csv << "," << v;
    for (double v : row.cellCapacity)state.csv << "," << v;
    state.csv << "," << row.servingSinrDb;
    for (double v : row.rsrpDbm)     state.csv << "," << v;
    state.csv << "," << row.targetBestCell << "\n";
}

static void
LogMobilityRow(RuntimeState&        state,
               const NodeContainer& enbNodes,
               const NodeContainer& ueNodes)
{
    if (!state.mobilityCsv.is_open())
        return;

    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        Ptr<MobilityModel> mob = ueNodes.Get(u)->GetObject<MobilityModel>();
        const Vector p = mob->GetPosition();
        const Vector v = mob->GetVelocity();

        state.mobilityCsv << std::fixed << std::setprecision(3)
                          << state.step << ","
                          << Simulator::Now().GetSeconds() << ","
                          << u << ","
                          << p.x << "," << p.y << ","
                          << v.x << "," << v.y << ","
                          << state.servingCell[u];

        for (uint32_t b = 0; b < enbNodes.GetN(); ++b)
        {
            const Vector ep =
                enbNodes.Get(b)->GetObject<MobilityModel>()->GetPosition();
            const double dx = p.x - ep.x;
            const double dy = p.y - ep.y;
            state.mobilityCsv << "," << std::sqrt(dx * dx + dy * dy);
        }
        state.mobilityCsv << "\n";
    }
}

static void
PrintFeatureMatrixSummary(const std::vector<FeatureRow>& rows, uint32_t numEnbs)
{
    NS_LOG_UNCOND("==============================================");
    NS_LOG_UNCOND("t=" << Simulator::Now().GetSeconds()
                  << "s  rows=" << rows.size());
    for (const auto& row : rows)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2)
            << "UE " << row.ueId
            << " pos=(" << row.ux << "," << row.uy << ")"
            << " serving=" << row.servingCell
            << " target="  << row.targetBestCell
            << " SINR[0]=" << row.sinrDb.at(0) << " dB"
            << " load[0]=" << row.cellLoad.at(0)
            << " cap[0]="  << row.cellCapacity.at(0) << " Mbps";
        NS_LOG_UNCOND(oss.str());
    }
    NS_LOG_UNCOND("==============================================");
}

// ===========================
// 19. NETANIM HELPERS
// ===========================
static void
ColorUesByServingCell(AnimationInterface*  anim,
                      const RuntimeState&  state,
                      const NodeContainer& ueNodes)
{
    if (!anim)
        return;
    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        const int32_t cell = state.servingCell[u];
        const uint8_t r = static_cast<uint8_t>((50  + 40 * (cell % 5)) % 255);
        const uint8_t g = static_cast<uint8_t>((80  + 70 * (cell % 3)) % 255);
        const uint8_t b = static_cast<uint8_t>((100 + 30 * (cell % 7)) % 255);
        anim->UpdateNodeDescription(ueNodes.Get(u),
                                    "UE-" + std::to_string(u) +
                                        "->eNB-" + std::to_string(cell));
        anim->UpdateNodeColor(ueNodes.Get(u), r, g, b);
    }
}

static std::unique_ptr<AnimationInterface>
SetupAnimation(const SimConfig& cfg,
               NodeContainer&   enbNodes,
               NodeContainer&   ueNodes)
{
    if (!cfg.enableNetAnim)
        return nullptr;

    auto anim = std::make_unique<AnimationInterface>(cfg.animFile);

    for (uint32_t i = 0; i < enbNodes.GetN(); ++i)
    {
        anim->UpdateNodeDescription(enbNodes.Get(i), "eNB-" + std::to_string(i));
        anim->UpdateNodeColor(enbNodes.Get(i), 255, 0, 0);
    }
    for (uint32_t i = 0; i < ueNodes.GetN(); ++i)
    {
        anim->UpdateNodeDescription(ueNodes.Get(i), "UE-" + std::to_string(i));
        anim->UpdateNodeColor(ueNodes.Get(i), 0, 0, 255);
    }
    return anim;
}

// ===========================
// 20. NS3-AI PYTHON INTERFACE
// ===========================
static std::vector<int32_t>
SendFeatureRowsToPython(const std::vector<FeatureRow>& rows,
                        const RuntimeState&            state)
{
    std::vector<int32_t> predictedCells(rows.size(), -1);

    if (!g_msgInterface)
    {
        NS_LOG_UNCOND("[ns3-ai] Interface null – falling back to targetBestCell.");
        for (uint32_t i = 0; i < rows.size(); ++i)
            predictedCells[i] = rows[i].targetBestCell;
        return predictedCells;
    }

    g_msgInterface->CppSendBegin();
    auto envVec = g_msgInterface->GetCpp2PyVector();
    envVec->resize(rows.size() + 1);

    for (uint32_t i = 0; i < rows.size(); ++i)
    {
        UserAssignmentEnv& env = envVec->at(i);
        env.done               = 0;
        env.timeSec            = rows[i].timeSec;
        env.ueId               = rows[i].ueId;
        env.numUes             = state.cfg.numUes;
        env.numEnbs            = state.cfg.numEnbs;
        env.ux                 = rows[i].ux;
        env.uy                 = rows[i].uy;
        env.vx                 = rows[i].vx;
        env.vy                 = rows[i].vy;
        env.reward             = state.lastReward;
        env.totalThroughputMbps= state.lastTotalThroughputMbps;
        env.loadStd            = state.lastLoadStd;
        env.handovers          = state.lastHandovers;
        env.outages            = state.lastOutages;
        env.step               = state.step;
        env.servingCell        = rows[i].servingCell;
        env.targetBestCell     = rows[i].targetBestCell;
        env.servingSinr        = rows[i].servingSinrDb;

        for (uint32_t b = 0; b < state.cfg.numEnbs; ++b)
        {
            env.load[b]     = rows[i].cellLoad[b];
            env.capacity[b] = rows[i].cellCapacity[b];
            env.rsrp[b]     = rows[i].rsrpDbm[b];
        }
    }

    NS_LOG_UNCOND("[ns3-ai] Sending " << rows.size()
                  << " rows to Python at t=" << Simulator::Now().GetSeconds() << "s");
    g_msgInterface->CppSendEnd();
    g_msgInterface->CppRecvBegin();

    auto actVec = g_msgInterface->GetPy2CppVector();

    if (actVec->size() < rows.size())
    {
        NS_LOG_UNCOND("[ns3-ai] Prediction vector too small – falling back to targetBestCell.");
        for (uint32_t i = 0; i < rows.size(); ++i)
            predictedCells[i] = rows[i].targetBestCell;
        g_msgInterface->CppRecvEnd();
        return predictedCells;
    }

    for (uint32_t i = 0; i < rows.size(); ++i)
    {
        int32_t pred = actVec->at(i).predictedCell;
        if (pred < 0 || pred >= static_cast<int32_t>(state.cfg.numEnbs))
        {
            NS_LOG_UNCOND("[ns3-ai] Invalid prediction for UE " << rows[i].ueId
                          << " – falling back to targetBestCell.");
            pred = rows[i].targetBestCell;
        }
        predictedCells[i] = pred;
    }

    g_msgInterface->CppRecvEnd();
    NS_LOG_UNCOND("[ns3-ai] Received predictions from Python.");
    return predictedCells;
}

static void
InitializeNs3AiInterface()
{
    auto interface = Ns3AiMsgInterface::Get();
    interface->SetIsMemoryCreator(false);
    interface->SetUseVector(true);
    interface->SetHandleFinish(true);
    g_msgInterface = interface->GetInterface<UserAssignmentEnv, UserAssignmentAct>();
    NS_LOG_UNCOND("[ns3-ai] Shared-memory interface initialized.");
}

// ===========================
// 21. PERIODIC ASSIGNMENT STEP
// ===========================
static void
AssignmentIntervalStep(NodeContainer  enbNodes,
                       NodeContainer  ueNodes,
                       RuntimeState*  state,
                       AnimationInterface* anim)
{
    // 1. Snapshot this interval's load fraction from real LTE scheduling bytes.
    //    cellCapacity is static and untouched here; this resets g_cellBytesScheduled.
    SnapshotCellStats(state->cfg.numEnbs,
                      state->cfg.assignmentInterval,
                      state->cellLoad,
                      state->cellCapacity);

    state->cellUserCounts = ComputeCellUserCounts(state->cfg, state->servingCell);

    // 2. Compute reward from real LTE data.
    //    GetUeThroughputMbps reads PacketSink byte delta and resets the baseline.
    const RewardMetrics rm = ComputeRewardMetrics(*state, ueNodes);
    state->lastReward              = rm.reward;
    state->lastTotalThroughputMbps = rm.totalThroughputMbps;
    state->lastLoadStd             = rm.loadStd;
    state->lastOutages             = rm.outages;

    NS_LOG_UNCOND("[RL] step="       << state->step
                  << " reward="      << state->lastReward
                  << " throughput="  << state->lastTotalThroughputMbps << " Mbps"
                  << " loadStd="     << state->lastLoadStd
                  << " handovers="   << state->lastHandovers
                  << " outages="     << state->lastOutages);

    // 3. Build feature rows from real LTE PHY/MAC measurements
    std::vector<FeatureRow> rows;
    rows.reserve(ueNodes.GetN());
    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        FeatureRow row = BuildFeatureRow(*state, ueNodes, u);
        rows.push_back(row);
        LogFeatureRow(*state, row);
    }

    // 4. Write per-UE positions, velocities, and distances to mobility CSV
    LogMobilityRow(*state, enbNodes, ueNodes);
    PrintFeatureMatrixSummary(rows, state->cfg.numEnbs);

    // 5. Send features to Python; receive RL cell assignments
    std::vector<int32_t> predictedCells = SendFeatureRowsToPython(rows, *state);

    // 6. Apply RL assignments via real LTE X2 handovers.
    //    Source cell must be the UE's REAL current serving cell (from PHY
    //    measurements), not the script's optimistic tracker: if a prior
    //    handover for this UE hasn't finished yet (RRC reconfig + random
    //    access at the target take time), the tracker can point at an eNB
    //    that never actually holds the UE's RRC context, and HandoverRequest
    //    would abort on GetUeManager() for that stale rnti/cell pair.
    uint32_t handoversThisStep = 0;
    for (const auto& row : rows)
    {
        const uint32_t u = row.ueId;
        const uint16_t realCellId = g_ueLteMeas[u].servingCellId;
        const int32_t oldCell =
            (realCellId > 0) ? static_cast<int32_t>(realCellId - 1) : state->servingCell[u];
        const int32_t newCell = predictedCells[u];

        // Skip if a previous handover for this UE hasn't completed yet.
        // Issuing a second HandoverRequest while the RRC is mid-procedure
        // (CONNECTED_HANDOVER) can catch the UE with a transiently invalid
        // RNTI, aborting on the "0 != rnti" assertion in LteEnbRrc.
        Ptr<LteUeRrc> ueRrc = g_ueDevs.Get(u)->GetObject<LteUeNetDevice>()->GetRrc();
        const bool handoverInProgress = ueRrc->GetState() != LteUeRrc::CONNECTED_NORMALLY;

        // Minimum dwell time since this UE's last handover. Without this,
        // a near-random exploring policy can re-target a UE every interval,
        // repeatedly tearing down/re-adding its spectrum listeners faster
        // than the PHY's interference bookkeeping settles (observed as a
        // "m_spectrumModel == x.m_spectrumModel" assert in spectrum-value.cc
        // under sustained rapid handover churn).
        const double nowSec = Simulator::Now().GetSeconds();
        const bool inCooldown =
            (nowSec - state->lastHandoverTimeSec[u]) < state->cfg.handoverCooldown;

        if (!handoverInProgress &&
            !inCooldown &&
            oldCell != newCell &&
            newCell >= 0 &&
            static_cast<uint32_t>(newCell) < state->cfg.numEnbs)
        {
            // X2 handover: LTE RRC + path-switch via EPC.
            // HandoverRequest's Time arg is a Simulator::Schedule() DELAY, not
            // an absolute time (despite the header doc) -- passing Now() here
            // would schedule the handover at 2x the current sim time, causing
            // requests to fire out of order relative to when they were issued.
            g_lteHelper->HandoverRequest(Seconds(0),
                                         g_ueDevs.Get(u),
                                         g_enbDevs.Get(oldCell),
                                         g_enbDevs.Get(newCell));
            ++handoversThisStep;
            state->servingCell[u] = newCell; // optimistic tracking update
            state->lastHandoverTimeSec[u] = nowSec;
        }
    }
    state->lastHandovers = handoversThisStep;
    state->step++;

    ColorUesByServingCell(anim, *state, ueNodes);

    // 7. Schedule next interval
    const double nextTime = Simulator::Now().GetSeconds() + state->cfg.assignmentInterval;
    if (nextTime <= state->cfg.simTime)
    {
        Simulator::Schedule(Seconds(state->cfg.assignmentInterval),
                            &AssignmentIntervalStep,
                            enbNodes, ueNodes, state, anim);
    }
}

// ===========================
// 22. DONE SIGNAL
// ===========================
static void
SendDoneSignalToPython()
{
    if (!g_msgInterface)
        return;
    g_msgInterface->CppSendBegin();
    auto envVec = g_msgInterface->GetCpp2PyVector();
    envVec->resize(1);
    envVec->at(0).done = 1;
    g_msgInterface->CppSendEnd();
    NS_LOG_UNCOND("[ns3-ai] Sent DONE signal to Python.");
}

// ===========================
// 23. MAIN
// ===========================
int
main(int argc, char* argv[])
{
    SimConfig cfg;
    ParseArgs(argc, argv, cfg);
    InitializeNs3AiInterface();

    // Nodes + mobility
    NodeContainer enbNodes, ueNodes;
    CreateNodes(cfg, enbNodes, ueNodes);
    InstallEnbMobility(cfg, enbNodes);
    InstallUeMobility(cfg, ueNodes);   // also sets UE z=1.5 m for Okumura-Hata

    // LTE stack (LteHelper + EPC + internet stack + IP routing)
    Ptr<Node>              remoteHost;
    Ipv4InterfaceContainer ueIpIface;
    SetupLteStack(cfg, enbNodes, ueNodes, remoteHost, ueIpIface);

    // X2 interface between all eNB pairs — required for HandoverRequest()
    g_lteHelper->AddX2Interface(enbNodes);

    // Initialise global LTE measurement arrays
    g_ueLteMeas.resize(cfg.numUes);
    for (auto& m : g_ueLteMeas)
        m.rsrpPerCellDbm.assign(cfg.numEnbs, -200.0); // -200 dBm = not yet measured
    g_cellBytesScheduled.assign(cfg.numEnbs, 0);
    g_ueBytesAtLastInterval.assign(cfg.numUes, 0);

    // Runtime state
    RuntimeState state;
    state.cfg           = cfg;
    state.servingCell   = InitialNearestCellAssignment(enbNodes, ueNodes);
    state.cellUserCounts= ComputeCellUserCounts(cfg, state.servingCell);
    state.cellLoad.assign(cfg.numEnbs, 0.0);
    // Static per-cell capacity allocation — set once, never rewritten per interval.
    state.cellCapacity.assign(cfg.numEnbs, LTE_PEAK_DL_MBPS);
    // -1e9 so every UE is immediately eligible for its first handover
    state.lastHandoverTimeSec.assign(cfg.numUes, -1e9);

    // Attach each UE to its nearest eNB (default EPS bearer set up automatically by EPC)
    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
        g_lteHelper->Attach(g_ueDevs.Get(u), g_enbDevs.Get(state.servingCell[u]));

    // Saturating UDP downlink traffic (remote host → each UE)
    SetupUdpTraffic(cfg, ueNodes, ueIpIface, remoteHost);

    // Wire up LTE PHY/MAC trace callbacks
    ConnectLteCallbacks(cfg, ueNodes, enbNodes);

    WriteCsvHeader(state);
    WriteMobilityCsvHeader(state, enbNodes);

    auto anim = SetupAnimation(cfg, enbNodes, ueNodes);
    ColorUesByServingCell(anim.get(), state, ueNodes);

    NS_LOG_UNCOND("Starting LTE user-assignment simulation");
    NS_LOG_UNCOND("Feature CSV:  " << cfg.featureCsv);
    NS_LOG_UNCOND("Mobility CSV: " << cfg.mobilityCsv);
    NS_LOG_UNCOND("NetAnim XML:  " << cfg.animFile);

    // First RL step at t=1.0 s — gives LTE connections time to establish
    // and RSRP/SINR measurement callbacks time to populate before first decision.
    Simulator::Schedule(Seconds(1.0),
                        &AssignmentIntervalStep,
                        enbNodes, ueNodes, &state, anim.get());

    Simulator::Stop(Seconds(cfg.simTime));
    Simulator::Run();

    SendDoneSignalToPython();
    Simulator::Destroy();

    if (state.csv.is_open())       state.csv.close();
    if (state.mobilityCsv.is_open())state.mobilityCsv.close();

    NS_LOG_UNCOND("Finished LTE user-assignment simulation");
    return 0;
}
