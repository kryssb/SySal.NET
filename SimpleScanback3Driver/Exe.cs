using System;
using SySal;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;
using SySal.DAQSystem.Drivers;
using System.Xml;
using System.Xml.Serialization;

namespace SySal.DAQSystem.Drivers.SimpleScanback3Driver
{
    class Prediction
    {
        public SySal.BasicTypes.Vector2 Pos;
        public SySal.BasicTypes.Vector2 Slope;
        public long Path;
        public long IdPath;
        public long IdCSEventBrick;
        public long IdCandidate;
        public long IdEvent;
        public int Track;
        public int Plate;
    }

    /// <summary>
    /// Direction of scanning procedure.
    /// </summary>
    [Serializable]
    public enum ScanDirection
    {
        /// <summary>
        /// The Scanback procedure starts from tracks in a downstream plate and traces them back to the production point going upstream.
        /// </summary>
        Upstream,
        /// <summary>
        /// The Scanforth procedure starts from tracks in an upstream plate and follows them downstream.
        /// </summary>
        Downstream
    }

    /// <summary>
    /// Source for Scanback/Scanforth path initialization.
    /// </summary>
    [Serializable]
    public enum PathSource
    {
        /// <summary>
        /// Paths are specified by an interrupt.
        /// </summary>
        Interrupt,
        /// <summary>
        /// Paths are initiated from the CS doublet.
        /// </summary>
        CSDoublet,
        /// <summary>
        /// Paths propagate reconstructed volume tracks downstream/upstream.
        /// </summary>
        VolumeTrack,
        /// <summary>
        /// Paths are initiated from predictions.
        /// </summary>
        Prediction,
        /// <summary>
        /// Paths are initiated from the CS doublet and connection information is filled.
        /// </summary>
        CSDoubletConnect
    }

    /// <summary>
    /// Settings for SimpleScanback3Driver.
    /// </summary>
    [Serializable]
    public class SimpleScanback3Settings
    {
        /// <summary>
        /// Configuration to be used for intercalibration of each plate.
        /// If this Id is equal to PredictionScanConfigId, the driver assumes that the same process operation performs intercalibration as well as scanning.
        /// If intercalibration is simply to be skipped, this Id should be set equal to PredictionScanConfigId, <b>not zero</b>.
        /// Ignored if <c>UseToRecomputeCalibrationsOnly</c> is set to <c>true</c>.
        /// </summary>
        public long IntercalibrationConfigId;
        /// <summary>
        /// Configuration to be used for PredictionScan.
        /// Ignored if <c>UseToRecomputeCalibrationsOnly</c> is set to <c>true</c>.
        /// </summary>
        public long PredictionScanConfigId;
        /// <summary>
        /// Configuration to be used for PredictionScan on the first plate. If this is zero, <c>PredictionScanConfigId</c> will be used.
        /// Ignored if <c>UseToRecomputeCalibrationsOnly</c> is set to <c>true</c>.
        /// </summary>
        public long FirstPlatePredictionScanConfigId;
        /// <summary>
        /// The maximum number of consecutive missing plates in a scanback/scanforth path. 
        /// If a scanback/scanforth track is not found consecutively for a number of plates that exceeds MaxMissingPlates, the path is terminated.
        /// Ignored if <c>UseToRecomputeCalibrationsOnly</c> is set to <c>true</c>.
        /// </summary>
        public int MaxMissingPlates;
        /// <summary>
        /// The number of plates to be skipped between two plates that are scanned. This is normally omitted or set to zero.
        /// </summary>
        public int SkipPlates;
        /// <summary>
        /// Direction of the scanning procedure if <c>UseToRecomputeCalibrationsOnly</c> is <c>false</c>.		
        /// <para><c>Upstream</c> designates Scanback, going from plates with high Z to plates with low Z.</para>
        /// <para><c>Downstream</c> designates Scanforth, going from plates with low Z to plates with high Z.</para>
        /// If <c>UseToRecomputeCalibrationsOnly</c> is set to <c>true</c>, the meaning becomes the following:
        /// <para><c>Upstream</c>: in each pair of plates, the upstream plate is calibrated w.r.t. the downstream one (therefore, intercalibration proceeds upstream).</para>
        /// <para><c>Downstream</c>: in each pair of plates, the downstream plate is calibrated w.r.t. the upstream one (therefore, intercalibration proceeds downstream).</para>
        /// </summary>
        public ScanDirection Direction;
        /// <summary>
        /// <para>
        /// This function, to be written in C-style, defines how the predicted position of a scanback paths is propagated from each plate to the next one. The function applies
        /// independently to each coordinate (X/Y). The variables that can be used are listed below:
        /// </para>
        /// <list type="table">
        /// <listheader><term>Name</term><description>Meaning</description></listheader>
        /// <item><term>LP</term><description>Last predicted position (X/Y).</description></item>
        /// <item><term>LS</term><description>Last predicted slope (X/Y).</description></item>
        /// <item><term>LZ</term><description>Reference Z of the last plate on which the path has been predicted.</description></item>
        /// <item><term>F</term><description>If a candidate has been found on the last plate, this field is <c>1</c>; <c>0</c> otherwise.</description></item>
        /// <item><term>FP</term><description>Last found position (X/Y). If no candidate has been found on the last plate, this variable is meaningless.</description></item>
        /// <item><term>FS</term><description>Last found slope (X/Y). If no candidate has been found on the last plate, this variable is meaningless.</description></item>
        /// <item><term>N</term><description>Number of grains of the last found candidate. If no candidate has been found, this variable is meaningless.</description></item>
        /// <item><term>A</term><description>Area sum of the last found candidate. If no candidate has been found, this variable is meaningless.</description></item>
        /// <item><term>S</term><description><c>Sigma</c> field of the last found candidate. If no candidate has been found, this variable is meaningless. <b>NOTICE: if the candidate is a weak base track (i.e. promoted microtrack), <c>Sigma</c> is negative; non-negative otherwise.</b></description></item>
        /// <item><term>Z</term><description>Reference Z of the plate for which predictions are to be produced.</description></item>
        /// </list>
        /// <para>A suggested propagation function is:</para>
        /// <code>(F == 0) * (LP + (Z - LZ) * LS) + (F == 1) * (FP + (Z - LZ) * ((S &lt; 0) * LS + (S &gt;= 0) * FS))</code>
        /// <para>The function does the following:
        /// <list type="bullet">
        /// <item>If no candidate has been found (<c>(F == 0)</c>), propagate the last predicted position using the last predicted slope;</item>
        /// <item>If a candidate has been found (<c>(F == 1)</c>), propagate the last found position using the last predicted slope if the candidate is weak (<c>S &lt; 0</c>) or the last found slope is the candidate is normal (<c>S &gt;= 0</c>).</item>
        /// </list>
        /// <b>NOTICE: when writing XML configurations manually, be aware that &gt; and &lt; are to be written as &amp;gt; and &amp;lt; respectively to avoid confusion with XML tag opening/closing marks.</b>
        /// </para>
        /// </summary>
        public string PositionPropagationFunction;
        /// <summary>
        /// <para>
        /// This function, to be written in C-style, defines how the predicted slope of a scanback paths is propagated from each plate to the next one. The function applies
        /// independently to each coordinate (X/Y). The variables that can be used are listed below:
        /// </para>
        /// <list type="table">
        /// <listheader><term>Name</term><description>Meaning</description></listheader>
        /// <item><term>LP</term><description>Last predicted position (X/Y).</description></item>
        /// <item><term>LS</term><description>Last predicted slope (X/Y).</description></item>
        /// <item><term>LZ</term><description>Reference Z of the last plate on which the path has been predicted.</description></item>
        /// <item><term>F</term><description>If a candidate has been found on the last plate, this field is <c>1</c>; <c>0</c> otherwise.</description></item>
        /// <item><term>FP</term><description>Last found position (X/Y). If no candidate has been found on the last plate, this variable is meaningless.</description></item>
        /// <item><term>FS</term><description>Last found slope (X/Y). If no candidate has been found on the last plate, this variable is meaningless.</description></item>
        /// <item><term>N</term><description>Number of grains of the last found candidate. If no candidate has been found, this variable is meaningless.</description></item>
        /// <item><term>A</term><description>Area sum of the last found candidate. If no candidate has been found, this variable is meaningless.</description></item>
        /// <item><term>S</term><description><c>Sigma</c> field of the last found candidate. If no candidate has been found, this variable is meaningless. <b>NOTICE: if the candidate is a weak base track (i.e. promoted microtrack), <c>Sigma</c> is negative; non-negative otherwise.</b></description></item>
        /// <item><term>Z</term><description>Reference Z of the plate for which predictions are to be produced.</description></item>
        /// </list>
        /// <para>A suggested propagation function is:</para>
        /// <code>(F == 0) * LS + (F == 1) * ((S &lt; 0) * LS + (S &gt;= 0) * FS)</code>
        /// <para>The function does the following:
        /// <list type="bullet">
        /// <item>If no candidate has been found (<c>(F == 0)</c>), use the last predicted slope;</item>
        /// <item>If a candidate has been found (<c>(F == 1)</c>), use the last predicted slope if the candidate is weak (<c>S &lt; 0</c>) or the last found slope is the candidate is normal (<c>S &gt;= 0</c>).</item>
        /// </list>
        /// </para>
        /// <b>NOTICE: when writing XML configurations manually, be aware that &gt; and &lt; are to be written as &amp;gt; and &amp;lt; respectively to avoid confusion with XML tag opening/closing marks.</b>
        /// </summary>
        public string SlopePropagationFunction;
        /// <summary>
        /// Source for Scanback/Scanforth paths.
        /// Ignored if <c>UseToRecomputeCalibrationsOnly</c> is set to <c>true</c>.
        /// </summary>		
        public PathSource Source;
        /// <summary>
        /// If <c>true</c>, calibrated plates are re-calibrated if they had been calibrated within a previous volume operation. Valid Calibrations are re-used if this is set to <c>false</c>.
        /// Ignored if <c>UseToRecomputeCalibrationsOnly</c> is set to <c>true</c>.
        /// </summary>
        public bool ForceRefreshCalibration;
        /// <summary>
        /// If <c>true</c>, SimpleScanback3Driver is used as a processing driver that computes refined intercalibration using scanback predictions from the latest SimpleScanback3Driver process on that brick.		
        /// </summary>
        public bool UseToRecomputeCalibrationsOnly;
        /// <summary>
        /// When <c>UseToRecomputeCalibrationsOnly</c> is <c>true</c>, this string is used in a <c>WHERE</c> clause to restrict scanback predictions to be used for recalibration. 
        /// If this string is empty or null, no restriction is applied.
        /// Ignored if <c>UseToRecomputeCalibrationsOnly</c> is <c>false</c>.
        /// </summary>
        public string WhereClauseForRecalibrationSelection;
        /// <summary>
        /// If set to <c>true</c>, the driver starts in <c>Halted</c> state, waiting for a <c>Continue</c> interrupt. This is useful when a full set of calibrations has to be forced/ignored, or when scanning must
        /// not start from the first plate. The flag is ignored if <c>UseToRecomputeCalibrationsOnly</c> is <c>false</c>.
        /// </summary>
        public bool WaitForContinue;
        /// <summary>
        /// If set to <c>true</c>, the driver does not close until a <c>Close</c> interrupt is sent. 
        /// </summary>
        /// <remarks>This function is useful to stay waiting for manually selected candidates, e.g. for CS-Brick connection.</remarks>
        public bool WaitForClose;
        /// <summary>
        /// If <c>true</c>, the projection DZ from CS to the most downstream plate is overridden.
        /// </summary>
        public bool CSDoubletDZOverride;
        /// <summary>
        /// The value to use for DZ if <c>CSDoubletDZOverride</c> is <c>true</c>; ignored otherwise.
        /// </summary>
        public double CSDoubletDZ;
        /// <summary>
        /// When no pre-selected candidates (TB_B_CSCANDS_SBPATHS) are found, this query is used to select candidates to propagate to brick. The number of the brick replaces any occurrence of  <c>_BRICK_</c>. The query must select rows in the format <c>ID_CS_EVENTBRICK, ID_CANDIDATE</c>.
        /// </summary>
        public string CSDoubletAutoSel;
        /// <summary>
        /// The number of plates to scan in the brick. Ignored if zero or negative. This field is useful for CS-Brick connection to limit the number of plates to be scanned (e.g., 5).
        /// </summary>
        public int PlatesToScan;
        /// <summary>
        /// If <c>true</c>, forked paths are not propagated. The default value (<c>false</c>) provides the standard behaviour, propagating all paths.
        /// </summary>
        public bool DoNotPropagateForkedPaths;
    }

    /// <summary>
    /// SimpleScanback3Driver executor.
    /// </summary>
    /// <remarks>
    /// <para>SimpleScanback3Driver performs scanback throughout a brick.</para>
    /// <para>Scanback paths cannot be forked.</para>
    /// <para>The TB_SCANBACK_PATHS and TB_SCANBACK_PREDICTIONS tables are used to record paths and their evolution.
    /// Predictions do not include tolerance specifications, which are therefore left entirely to the lower-level driver that scans the plates.</para>
    /// <para>At each plate, the prediction is updated using the last found candidate for both position and slopes.</para>
    /// <para>Position tolerance is not expanded if the candidate is not found on one or more plates.</para>
    /// <para>Type: <c>SimpleScanback3Driver /Interrupt &lt;batchmanager&gt; &lt;process operation id&gt; &lt;interrupt string&gt;</c> to send an interrupt message to a running SimpleScanback3Driver process operation.</para>
    /// <para>
    /// Supported Interrupts:
    /// <list type="table">
    /// <item><term><c>Continue</c></term><description>If <c>WaitForContinue</c> is set in the program settings, this interrupt is required to actually start the task.</description></item>
    /// <item><term><c>Close</c></term><description>If <c>WaitForClose</c> is set in the program settings, this interrupt is required to actually close the task; when waiting for this interrupt, the process cannot be "rolled back" to previous plates.</description></item>
    /// <item><term><c>PlateDamagedCode &lt;code&gt;</c></term><description>Instructs SimpleScanback3Driver to use the specified code to mark the plate as damaged. The plate must be specified by PlateDamaged. The plate damaged code is a single character; <c>N</c> means no damage.</description></item>
    /// <item><term><c>PlateDamaged &lt;plate&gt;</c></term><description>Instructs SimpleScanback3Driver to mark the specified plate as damaged. If it is missing, the current plate number is assumed.</description></item>
    /// <item><term><c>GoBackToPlateN &lt;plate&gt;</c></term><description>Instructs SimpleScanback3Driver to go back to the specified plate, keeping intercalibration info. The plate specified is the first plate for which predictions as well as results will be kept.</description></item>
    /// <item><term><c>GoBackToPlateNCancelCalibrations &lt;plate&gt;</c></term><description>Instructs SimpleScanback3Driver to go back to the specified plate, cancelling intercalibration info obtained by daughter operations of the current one. Intercalibrations obtained by previous operations will not be cancelled. The plate specified is the first plate for which predictions as well as results will be kept.</description></item>
    /// <item><term><c>Paths &lt;number&gt;</c></term><description>Used to provide predictions on startup when the path source is set to Interrupt in the program settings. <c>number</c> sets the number of expected predictions. Predictions are 6-tuples such as <c>PATH POSX POSY SLOPEX SLOPEY FIRSTPLATE</c> separated by ';'. The first prediction must be preceeded by ';'. No ';' is to be put at the end of the prediction string. The prediction string may contain any spacers including newlines.</description></item>
    /// <item><term><c>IgnoreCalibrations &lt;number&gt;</c></term><description>Used to provide a list of ignored calibrations. <c>number</c> sets the number of ignored calibrations. Each calibration is identified with its ID (ID_PROCESSOPERATION in TB_PLATE_CALIBRATIONS), and IDs must be separated by any spacer character.</description></item>
    /// <item><term><c>ForceCalibrations &lt;number&gt;</c></term><description>Used to provide a list of forced calibrations. <c>number</c> sets the number of forced calibrations. Each calibration is identified with its ID (ID_PROCESSOPERATION in TB_PLATE_CALIBRATIONS), and IDs must be separated by any spacer character.</description></item>
    /// <item><term><c>SetPaths &lt;number&gt;</c></term><description>Used to allow manual setting of path candidates. <c>number</c> sets the number of rows following. Each row has the format <c>PATH ZONE CANDBASE CANDUPMICRO CANDDOWNMICRO</c>. If CANDBASE is greater than zero, the candidate is set to the corresponding base track; if CANDBASE is zero or less, the candidate is formed by creating a new base track made with the microtracks specified by CANDDOWN and CANDUP, if they are both greater than zero; 
    /// if only one of them is greater than zero, the candidate is set to a promoted microtrack; if CANDBASE=CANDDOWN=CANDUP=0, the candidate is reset to <c>NULL</c>.</description></item>
    /// </list>
    /// </para>
    /// <para>An example of interrupt for path specification follows:</para>
    /// <para>
    /// <example>
    /// <code>
    /// Paths 3;
    /// 1 10204.3 14893.2 0.238 0.008 1;
    /// 2 11244.3 18823.2 -0.182 0.080 5;
    /// 3 8848.1 1248.5 -0.006 0.185 2
    /// </code>
    /// </example>
    /// </para>
    /// <para>Type: <c>SimpleScanback3Driver /EasyInterrupt</c> for a graphical user interface to send interrupts.</para>
    /// <para>
    /// A sample XML configuration for SimpleScanback3Driver follows:
    /// <example>
    /// <code>
    /// &lt;SimpleScanback3Settings&gt;
    ///  &lt;IntercalibrationConfigId&gt;1000000001570150&lt;/IntercalibrationConfigId&gt;
    ///  &lt;PredictionScanConfigId&gt;1000000001587818&lt;/PredictionScanConfigId&gt;
    ///  &lt;MaxMissingPlates&gt;3&lt;/MaxMissingPlates&gt;
    ///  &lt;Direction&gt;Upstream&lt;/Direction&gt;
    /// &lt;/SimpleScanback3Settings&gt;
    /// </code>
    /// </example>
    /// </para>
    /// </remarks>
    public class Exe : MarshalByRefObject, IInterruptNotifier, SySal.Web.IWebApplication
    {
        /// <summary>
        /// Initializes the Lifetime Service.
        /// </summary>
        /// <returns>the lifetime service object or null.</returns>
        public override object InitializeLifetimeService()
        {
            return null;
        }

        static void ShowExplanation()
        {
            ExplanationForm EF = new ExplanationForm();
            System.IO.StringWriter strw = new System.IO.StringWriter();
            strw.WriteLine("");
            strw.WriteLine("SimpleScanback3Driver");
            strw.WriteLine("--------------");
            strw.WriteLine("SimpleScanback3Driver performs scanback throughout a brick.");
            strw.WriteLine("Scanback paths cannot be forked.");
            strw.WriteLine("The TB_SCANBACK_PATHS and TB_SCANBACK_PREDICTIONS tables are used to record paths and their evolution.");
            strw.WriteLine("Predictions do not include tolerance specifications, which are therefore left entirely to the lower-level driver that scans the plates.");
            strw.WriteLine("At each plate, the prediction is updated using the last found candidate for both position and slopes.");
            strw.WriteLine("Position tolerance is not expanded if the candidate is not found on one or more plates.");
            strw.WriteLine("--------------");
            strw.WriteLine("Scanback initialization comes from the startup file. The BottomPlate and TopPlate must both be set equal to the initial plate.");
            strw.WriteLine("The TB_SCANBACK_PREDICTIONS and TB_SCANBACK_PATHS tables are used to record the scanback process.");
            strw.WriteLine();
            strw.WriteLine("Type: SimpleScanback3Driver /Interrupt <batchmanager> <process operation id> <interrupt string>");
            strw.WriteLine("to send an interrupt message to a running SimpleScanback3Driver process operation.");
            strw.WriteLine("SUPPORTED INTERRUPTS:");
            strw.WriteLine("PlateDamagedCode <code> - Instructs SimpleScanback3Driver to use the specified code to mark the plate as damaged. The plate must be specified by PlateDamaged. The plate damaged code is a single character; 'N' means no damage.");
            strw.WriteLine("PlateDamaged <plate> - Instructs SimpleScanback3Driver to mark the specified plate as damaged. If it is missing, the current plate number is assumed.");
            strw.WriteLine("GoBackToPlateN <plate> - Instructs SimpleScanback3Driver to go back to the specified plate, keeping intercalibration info. The plate specified is the first plate for which predictions as well as results will be kept.");
            strw.WriteLine("GoBackToPlateNCancelCalibrations <plate> - Instructs SimpleScanback3Driver to go back to the specified plate, cancelling intercalibration info obtained by daughter operations of the current one. Intercalibrations obtained by previous operations will not be cancelled. The plate specified is the first plate for which predictions as well as results will be kept.");
            strw.WriteLine("Paths <number> - Used to provide predictions on startup when the path source is set to Interrupt in the program settings. 'Number' sets the number of expected predictions. Predictions are 6-tuples such as PATH POSX POSY SLOPEX SLOPEY FIRSTPLATE separated by ';'. The first prediction must be preceeded by ';'. No ';' is to be put at the end of the prediction string. The prediction string may contain any spacers including newlines.");
            strw.WriteLine("IgnoreCalibrations <number> - Used to provide a list of ignored calibrations. 'Number' sets the number of ignored calibrations. Each calibration is identified with its ID (ID_PROCESSOPERATION in TB_PLATE_CALIBRATIONS), and IDs must be separated by any spacer character.");
            strw.WriteLine("ForceCalibrations <number> - Used to provide a list of forced calibrations. 'Number' sets the number of forced calibrations. Each calibration is identified with its ID (ID_PROCESSOPERATION in TB_PLATE_CALIBRATIONS), and IDs must be separated by any spacer character.");
            strw.WriteLine("Type: SimpleScanback3Driver /EasyInterrupt for a graphical user interface to send interrupts.");
            strw.WriteLine("--------------");
            strw.WriteLine("An example of interrupt for path specification follows:");
            strw.WriteLine("Paths 3;");
            strw.WriteLine("1 10204.3 14893.2 0.238 0.008 1;");
            strw.WriteLine("2 11244.3 18823.2 -0.182 0.080 5;");
            strw.WriteLine("3 8848.1 1248.5 -0.006 0.185 2");
            strw.WriteLine("--------------");
            strw.WriteLine("The program settings should have the following structure:");
            SimpleScanback3Settings sbset = new SimpleScanback3Settings();
            sbset.IntercalibrationConfigId = 1000000001570150;
            sbset.PredictionScanConfigId = 1000000001587818;
            sbset.MaxMissingPlates = 3;
            sbset.Direction = ScanDirection.Upstream;
            new System.Xml.Serialization.XmlSerializer(typeof(SimpleScanback3Settings)).Serialize(strw, sbset);
            EF.RTFOut.Text = strw.ToString();
            EF.ShowDialog();
        }
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [MTAThread]
        internal static void Main(string[] args)
        {
            HE = SySal.DAQSystem.Drivers.HostEnv.Own;
            if (HE == null)
            {
                if (args.Length == 4 && String.Compare(args[0].Trim(), "/Interrupt", true) == 0) SendInterrupt(args[1], Convert.ToInt64(args[2]), args[3]);
                else if (args.Length == 1 && String.Compare(args[0].Trim(), "/EasyInterrupt", true) == 0) EasyInterrupt();
                else ShowExplanation();
                return;
            }

            Execute();
        }

        private static void SendInterrupt(string machine, long op, string interruptdata)
        {
            SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
            SySal.OperaDb.OperaDbConnection conn = cred.Connect();
            conn.Open();
            string addr = new SySal.OperaDb.OperaDbCommand("SELECT ADDRESS FROM TB_MACHINES WHERE ID = '" + machine + "' OR UPPER(NAME) = '" + machine.ToUpper() + "' OR UPPER(ADDRESS) = '" + machine.ToUpper() + "' AND ISBATCHSERVER <> 0", conn).ExecuteScalar().ToString();
            Console.WriteLine("Contacting Batch Manager " + addr);
            ((SySal.DAQSystem.BatchManager)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.BatchManager), "tcp://" + addr + ":" + ((int)SySal.DAQSystem.OperaPort.BatchServer) + "/BatchManager.rem")).Interrupt(op, cred.OPERAUserName, cred.OPERAPassword, interruptdata);
        }

        private static System.Threading.ManualResetEvent CanRun = new System.Threading.ManualResetEvent(true);

        private static System.Threading.ManualResetEvent CanEnd = new System.Threading.ManualResetEvent(true);

        private static SySal.DAQSystem.Drivers.HostEnv HE = null;

        private static SySal.OperaDb.OperaDbConnection Conn;

        private static SimpleScanback3Settings ProgSettings;

        private static SySal.DAQSystem.Drivers.VolumeOperationInfo StartupInfo;

        private static SySal.DAQSystem.Drivers.TaskProgressInfo ProgressInfo = null;

        private static bool PredictionsDone;

        private static bool IntercalibrationDone;

        private static bool ScanDone;

        private static int Plate;

        private static long WaitingOnId;

        private static long CalibrationId;

        private static bool ReloadStatus = false;

        private static int MinPlate, MaxPlate;

        private static double MinZ, MaxZ;

        private static System.Drawing.Bitmap gIm = null;

        private static System.Drawing.Graphics gMon = null;

        private static NumericalTools.Plot gPlot = null;

        private static System.Exception PredException = null;

        private static System.Threading.ManualResetEvent PredictionsReceivedEvent = new System.Threading.ManualResetEvent(false);

        private static System.Threading.ManualResetEvent PredictionsDoneEvent = new System.Threading.ManualResetEvent(false);

        private static Prediction[] PredictionsReceived = null;

        private static bool WaitForPredictions = false;

        private static bool NoMorePredictions = false;

        private static int NextPredictedPlate = -1;

        private static NumericalTools.CStyleParsedFunction PositionPropagationF;

        private static NumericalTools.CStyleParsedFunction SlopePropagationF;

        private static System.Threading.Thread ThisThread = null;

        private static System.Threading.Thread WorkerThread = new System.Threading.Thread(new System.Threading.ThreadStart(WorkerThreadExec));

        private static System.Collections.Queue WorkQueue = new System.Collections.Queue();

        private static void WorkerThreadExec()
        {
            try
            {
                while (true)
                {
                    int qc;
                    try
                    {
                        System.Threading.Thread.Sleep(System.Threading.Timeout.Infinite);
                    }
                    catch (System.Threading.ThreadInterruptedException) { }
                    lock (WorkQueue)
                        if ((qc = WorkQueue.Count) == 0) return;
                    while (qc > 0)
                    {
                        lock (WorkQueue)
                        {
                            WorkQueue.Dequeue();
                            qc = WorkQueue.Count;
                        }
                        MakePredictions();
                    }
                }
            }
            catch (System.Threading.ThreadAbortException)
            {
                System.Threading.Thread.ResetAbort();
            }
            catch (Exception) { }
        }

        private static System.Collections.ArrayList IgnoreCalibrationList = new System.Collections.ArrayList();

        private static System.Collections.ArrayList ForceCalibrationList = new System.Collections.ArrayList();

        private static void EasyInterrupt()
        {
            (new frmEasyInterrupt()).ShowDialog();
        }

        private static void UpdateProgress()
        {
            ProgressInfo.CustomInfo = "\t\t[Plate] " + Plate.ToString() + " [/Plate]\r\n\t\t[WaitingOnId] " + WaitingOnId.ToString() + " [/WaitingOnId]\r\n\t\t[CalibrationId] " + CalibrationId.ToString() + " [/CalibrationId]\r\n\t\t[IntercalibrationDone] " + IntercalibrationDone + " [/IntercalibrationDone]\r\n\t\t[PredictionsDone] " + PredictionsDone + " [/PredictionsDone]\r\n\t\t[ScanDone] " + ScanDone + " [/ScanDone]\r\n\t\t[IgnoreCalibrationList]\r\n";
            foreach (long ic in IgnoreCalibrationList)
                ProgressInfo.CustomInfo += "\t\t\t\r\n[long]" + ic.ToString() + "[/long]\r\n";
            ProgressInfo.CustomInfo += "\t\t[/IgnoreCalibrationList]\r\n\t\t[ForceCalibrationList]\r\n";
            foreach (long ic in ForceCalibrationList)
                ProgressInfo.CustomInfo += "\t\t\t\r\n[long]" + ic.ToString() + "[/long]\r\n";
            ProgressInfo.CustomInfo += "\t\t[/ForceCalibrationList]\r\n";
            HE.ProgressInfo = ProgressInfo;
        }

        private static void UpdatePlots()
        {
            System.IO.StreamWriter w = null;
            if ((ProgSettings.Source == PathSource.Interrupt && ((ProgSettings.Direction == ScanDirection.Upstream && Plate == MaxPlate) || (ProgSettings.Direction == ScanDirection.Downstream && Plate == MinPlate))))
            {
                try
                {
                    w = new System.IO.StreamWriter(StartupInfo.ScratchDir + "\\" + StartupInfo.ProcessOperationId + "_progress.htm");
                    w.WriteLine(
                        "<html><head>" + (((ProgSettings.Direction == ScanDirection.Upstream && Plate < MaxPlate) || (ProgSettings.Direction == ScanDirection.Downstream && Plate > MinPlate)) ? "<meta http-equiv=\"REFRESH\" content=\"60\">" : "") + "<title>SimpleScanback3Driver Monitor</title></head><body>\r\n" +
                        "<div align=center><p><font face=\"Arial, Helvetica\" size=4 color=4444ff>SimpleScanback3Driver Brick #" + StartupInfo.BrickId + "<br>Operation ID = " + StartupInfo.ProcessOperationId + "</font><hr></p></div>\r\n" +
                        "<div align=center><p><font face = \"Arial, Helvetica\" size=6 color=FF3333><b>Waiting for prediction list from interrupt</b></font></p></div>\r\n" +
                        "</body></html>"
                        );
                    w.Flush();
                }
                catch (Exception) { }
                finally
                {
                    if (w != null) w.Close();
                }
            }
            else
            {
                Conn.Open();
                System.Data.DataSet dspred = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT /*+INDEX_ASC (TB_SCANBACK_PREDICTIONS PK_SCANBACK_PREDICTIONS) */ ID_PLATE, COUNT(*) AS NPRED FROM TB_SCANBACK_PREDICTIONS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PATH IN (SELECT /*+INDEX_ASC (TB_SCANBACK_PATHS PK_SCANBACK_PATHS) */ ID FROM TB_SCANBACK_PATHS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PROCESSOPERATION = " + StartupInfo.ProcessOperationId + ") GROUP BY ID_PLATE ORDER BY ID_PLATE ASC", Conn, null).Fill(dspred);
                System.Data.DataSet dsfound = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT /*+INDEX_ASC (TB_SCANBACK_PREDICTIONS PK_SCANBACK_PREDICTIONS) */ ID_PLATE, COUNT(*) AS NFOUND FROM TB_SCANBACK_PREDICTIONS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PATH IN (SELECT /*+INDEX_ASC (TB_SCANBACK_PATHS PK_SCANBACK_PATHS) */ ID FROM TB_SCANBACK_PATHS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PROCESSOPERATION = " + StartupInfo.ProcessOperationId + ") AND ID_CANDIDATE IS NOT NULL GROUP BY ID_PLATE ORDER BY ID_PLATE ASC", Conn, null).Fill(dsfound);
                System.Data.DataSet dsfill = new System.Data.DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_PATH, NFOUND, MINPLATE, MAXPLATE, COUNT(*) AS GOODPLATES FROM (SELECT /*+INDEX_ASC (TB_SCANBACK_PREDICTIONS PK_SCANBACK_PREDICTIONS) */ ID_PATH, COUNT(*) AS NFOUND, MIN(ID_PLATE) AS MINPLATE, MAX(ID_PLATE) AS MAXPLATE FROM TB_SCANBACK_PREDICTIONS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PATH IN (SELECT /*+INDEX_ASC (TB_SCANBACK_PATHS PK_SCANBACK_PATHS) */ TB_SCANBACK_PATHS.ID FROM TB_SCANBACK_PATHS INNER JOIN TB_SCANBACK_PREDICTIONS ON (TB_SCANBACK_PATHS.ID_EVENTBRICK = TB_SCANBACK_PREDICTIONS.ID_EVENTBRICK AND TB_SCANBACK_PATHS.ID = TB_SCANBACK_PREDICTIONS.ID_PATH) WHERE TB_SCANBACK_PATHS.ID_EVENTBRICK = " + StartupInfo.BrickId + " AND TB_SCANBACK_PATHS.ID_PROCESSOPERATION = " + StartupInfo.ProcessOperationId + " AND TB_SCANBACK_PREDICTIONS.ID_PLATE = " + Plate + ") AND ID_CANDIDATE IS NOT NULL GROUP BY ID_PATH) INNER JOIN VW_PLATES ON (VW_PLATES.ID >= MINPLATE AND VW_PLATES.ID <= MAXPLATE AND DAMAGED = 'N' AND ID_EVENTBRICK = " + StartupInfo.BrickId + ") GROUP BY ID_PATH, NFOUND, MINPLATE, MAXPLATE", Conn, null).Fill(dsfill);
                Conn.Close();

                int i, initial = 0, current = 0;
                double[] plates;
                double[] counts;
                plates = new double[dspred.Tables[0].Rows.Count];
                counts = new double[dspred.Tables[0].Rows.Count];
                try
                {
                    for (i = 0; i < plates.Length; i++)
                    {
                        plates[i] = Convert.ToDouble(dspred.Tables[0].Rows[i][0]);
                        counts[i] = Convert.ToDouble(dspred.Tables[0].Rows[i][1]);
                    }
                    initial = Convert.ToInt32(Math.Max(counts[0], counts[counts.Length - 1]));
                    current = Convert.ToInt32(Math.Min(counts[0], counts[counts.Length - 1]));
                    if (plates.Length > 1)
                    {
                        gMon.Clear(System.Drawing.Color.White);
                        gPlot.VecX = plates;
                        gPlot.SetXDefaultLimits = false;
                        gPlot.DX = 1.0f;
                        gPlot.MinX = plates[0] - 1;
                        gPlot.MaxX = plates[plates.Length - 1] + 1;
                        gPlot.XTitle = "Plate #";
                        gPlot.VecY = counts;
                        gPlot.SetYDefaultLimits = false;
                        gPlot.MinY = 0.0;
                        gPlot.MaxY = Math.Max(counts[0], counts[counts.Length - 1]) + 1.0;
                        gPlot.DY = (float)(gPlot.MaxY * 0.1);
                        gPlot.YTitle = "Predictions";
                        gPlot.PanelFormat = "F0";
                        gPlot.PanelX = 0.5;
                        gPlot.PanelY = 0.0;
                        //gPlot.HistoFit = 0;
                        gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                        gIm.Save(StartupInfo.ScratchDir + "\\" + StartupInfo.ProcessOperationId + "_00.png", System.Drawing.Imaging.ImageFormat.Png);
                    }
                }
                catch (Exception) { }
                try
                {
                    plates = new double[dsfound.Tables[0].Rows.Count];
                    counts = new double[dsfound.Tables[0].Rows.Count];
                    for (i = 0; i < plates.Length; i++)
                    {
                        plates[i] = Convert.ToDouble(dsfound.Tables[0].Rows[i][0]);
                        counts[i] = Convert.ToDouble(dsfound.Tables[0].Rows[i][1]);
                    }
                    if (plates.Length > 1)
                    {
                        gMon.Clear(System.Drawing.Color.White);
                        gPlot.VecX = plates;
                        gPlot.SetXDefaultLimits = false;
                        gPlot.DX = 1.0f;
                        gPlot.MinX = plates[0] - 1;
                        gPlot.MaxX = plates[plates.Length - 1] + 1;
                        gPlot.XTitle = "Plate #";
                        gPlot.VecY = counts;
                        gPlot.SetYDefaultLimits = false;
                        gPlot.MinY = 0.0;
                        gPlot.MaxY = Math.Max(counts[0], counts[counts.Length - 1]) + 1.0;
                        gPlot.DY = (float)(gPlot.MaxY * 0.1);
                        gPlot.YTitle = "Found";
                        gPlot.PanelFormat = "F0";
                        gPlot.PanelX = 0.5;
                        gPlot.PanelY = 0.0;
                        //gPlot.HistoFit = 0;
                        gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                        gIm.Save(StartupInfo.ScratchDir + "\\" + StartupInfo.ProcessOperationId + "_01.png", System.Drawing.Imaging.ImageFormat.Png);
                    }
                }
                catch (Exception) { }

                try
                {
                    double[] fillfactor = new double[dsfill.Tables[0].Rows.Count];
                    for (i = 0; i < fillfactor.Length; i++)
                        //fillfactor[i] = Convert.ToDouble(dsfill.Tables[0].Rows[i][1]) / (Convert.ToDouble(dsfill.Tables[0].Rows[i][3]) - Convert.ToDouble(dsfill.Tables[0].Rows[i][2]) + 1.0) * 100.0;
                        fillfactor[i] = Convert.ToDouble(dsfill.Tables[0].Rows[i][1]) / Convert.ToDouble(dsfill.Tables[0].Rows[i][4]) * 100.0;
                    gMon.Clear(System.Drawing.Color.White);
                    gPlot.VecX = fillfactor;
                    gPlot.SetXDefaultLimits = false;
                    gPlot.DX = 5.0f;
                    gPlot.MinX = 0.0f;
                    gPlot.MaxX = 105.0f;
                    gPlot.XTitle = "Fill factor";
                    gPlot.SetYDefaultLimits = true;
                    gPlot.PanelFormat = "F2";
                    gPlot.PanelX = 0.5;
                    gPlot.PanelY = 0.0;
                    gPlot.HistoFit = -2;
                    gPlot.Histo(gMon, gIm.Width, gIm.Height);
                    gIm.Save(StartupInfo.ScratchDir + "\\" + StartupInfo.ProcessOperationId + "_02.png", System.Drawing.Imaging.ImageFormat.Png);
                }
                catch (Exception) { }
                try
                {
                    w = new System.IO.StreamWriter(StartupInfo.ScratchDir + "\\" + StartupInfo.ProcessOperationId + "_progress.htm");
                    w.WriteLine(
                        "<html><head>" + (((ProgSettings.Direction == ScanDirection.Upstream && Plate < MaxPlate) || (ProgSettings.Direction == ScanDirection.Downstream && Plate > MinPlate)) ? "<meta http-equiv=\"REFRESH\" content=\"60\">" : "") + "<title>SimpleScanback3Driver Monitor</title></head><body>\r\n" +
                        "<div align=center><p><font face=\"Arial, Helvetica\" size=4 color=4444ff>SimpleScanback3Driver Brick #" + StartupInfo.BrickId + "<br>Operation ID = " + StartupInfo.ProcessOperationId + "</font><hr></p></div>\r\n" +
                        "<div align=center><p><font face = \"Arial, Helvetica\" size=2 color=0000cc>Initial = " + initial + "<br>Current = " + current + " (" + ((double)current / (double)initial * 100.0).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "%)</font></p>\r\n" +
                        "<table border=1 align=center>\r\n" +
                        "<tr><td><img src=\"" + StartupInfo.ProcessOperationId + "_00.png\" border=0></td></tr>\r\n" +
                        "<tr><td><img src=\"" + StartupInfo.ProcessOperationId + "_01.png\" border=0></td></tr>\r\n" +
                        "<tr><td><img src=\"" + StartupInfo.ProcessOperationId + "_02.png\" border=0></td></tr>\r\n" +
                        "</table></div></body></html>"
                        );
                    w.Flush();
                }
                catch (Exception) { }
                finally
                {
                    if (w != null) w.Close();
                }
            }
        }

        private delegate void dMakePredictions();

        private static void MakePredictions()
        {
            // Second step: Predictions
            SySal.OperaDb.OperaDbTransaction trans = null;
            try
            {
                //Conn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
                Conn.Open();
                SySal.OperaDb.Schema.DB = Conn;
                /*
                                if (SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM TB_SCANBACK_PREDICTIONS WHERE (ID_EVENTBRICK, ID_PLATE, ID_PATH) IN (SELECT " + StartupInfo.BrickId + ", " + Plate + ", ID FROM TB_SCANBACK_PATHS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PROCESSOPERATION = " + StartupInfo.ProcessOperationId + ")", Conn, null).ExecuteScalar()) > 0)
                                {
                                    HE.WriteLine("Predictions found already in place.");
                                    PredictionsDone = true;
                                    PredictionsDoneEvent.Set();
                                    UpdateProgress();
                                    return;
                                }
                 */
                if (PredictionsDone == false)
                {
                    const int BatchSize = 100;
                    int i, b;
                    System.Collections.ArrayList preds = new System.Collections.ArrayList();
                    bool isbrickentrypoint = false;
                    int csdoubletselcands = 0;

                    trans = Conn.BeginTransaction();
                    HE.WriteLine("Plate " + Plate + " MinPlate " + MinPlate + " MaxPlate " + MaxPlate + " Direction " + ProgSettings.Direction);
                    if ((Plate == MaxPlate && ProgSettings.Direction == SimpleScanback3Driver.ScanDirection.Upstream) ||
                        (Plate == MinPlate && ProgSettings.Direction == SimpleScanback3Driver.ScanDirection.Downstream))
                    {
                        isbrickentrypoint = true;
                        HE.WriteLine("Brick entry point");
                        if (ProgSettings.Source == PathSource.Interrupt)
                        {
                            long[] a_series = new long[BatchSize];
                            long[] a_idpath = new long[BatchSize];

                            Prediction[] predsread = null;
                            PredictionsReceivedEvent.WaitOne();
                            HE.WriteLine("Predictions received");
                            predsread = PredictionsReceived;
                            PredictionsReceived = null;

                            for (i = 0; i < predsread.Length; i++) preds.Add(predsread[i]);

                            lock (Conn)
                            {
                                for (i = 0; i < preds.Count; i++)
                                {
                                    Prediction p = (Prediction)preds[i];
                                    p.IdPath = SySal.OperaDb.Schema.TB_SCANBACK_PATHS.Insert(StartupInfo.BrickId, StartupInfo.ProcessOperationId, p.IdPath, 0, p.Plate, System.DBNull.Value, System.DBNull.Value);
                                }
                            }
                            HE.WriteLine("Plate #" + Plate + " - Injected #" + preds.Count + " paths.");
                        }
                        else if (ProgSettings.Source == PathSource.Prediction)
                        {
                            SySal.BasicTypes.Vector BrickZero = new SySal.BasicTypes.Vector();
                            System.Data.DataSet dsbz = new System.Data.DataSet();
                            new SySal.OperaDb.OperaDbDataAdapter("SELECT ZEROX, ZEROY, ZEROZ FROM TB_EVENTBRICKS WHERE ID = " + StartupInfo.BrickId, Conn, null).Fill(dsbz);
                            BrickZero.X = SySal.OperaDb.Convert.ToDouble(dsbz.Tables[0].Rows[0][0]);
                            BrickZero.Y = SySal.OperaDb.Convert.ToDouble(dsbz.Tables[0].Rows[0][1]);
                            BrickZero.Z = SySal.OperaDb.Convert.ToDouble(dsbz.Tables[0].Rows[0][2]);

                            int pathid = 0;
                            double ZTarget;
                            ZTarget = (ProgSettings.Direction == ScanDirection.Upstream) ? MaxZ : MinZ;

                            System.Data.DataSet dps = new System.Data.DataSet();
                            new SySal.OperaDb.OperaDbDataAdapter("SELECT /*+INDEX(TB_PREDICTED_TRACKS PK_PREDICTED_TRACKS) */ ID_EVENT, TRACK, POSX, POSY, POSZ, SLOPEX, SLOPEY FROM TB_PREDICTED_TRACKS WHERE (ID_EVENT, TRACK) IN " +
                                "(" +
                                " SELECT /*+INDEX_ASC(TB_B_PREDTRACKS_SBPATHS IX_PREDTRACKS_SBPATHS_INPUTS) */ ID_EVENT, TRACK FROM TB_B_PREDTRACKS_SBPATHS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_SCANBACK_PROCOPID IS NULL AND PATH IS NULL" +
                                ")", Conn, null).Fill(dps);

                            System.Data.DataRowCollection drc = dps.Tables[0].Rows;
                            foreach (System.Data.DataRow dr in drc)
                            {
                                Prediction p = new Prediction();
                                p.Path = ++pathid;
                                p.Slope.X = SySal.OperaDb.Convert.ToDouble(dr[5]);
                                p.Slope.Y = SySal.OperaDb.Convert.ToDouble(dr[6]);
                                double z = SySal.OperaDb.Convert.ToDouble(dr[4]);
                                p.Pos.X = SySal.OperaDb.Convert.ToDouble(dr[2]) + p.Slope.X * (ZTarget - z) - BrickZero.X;
                                p.Pos.Y = SySal.OperaDb.Convert.ToDouble(dr[3]) + p.Slope.Y * (ZTarget - z) - BrickZero.Y;
                                p.IdEvent = SySal.OperaDb.Convert.ToInt64(dr[0]);
                                p.Track = SySal.OperaDb.Convert.ToInt32(dr[1]);
                                p.Plate = Plate;
                                p.IdPath = SySal.OperaDb.Schema.TB_SCANBACK_PATHS.Insert(StartupInfo.BrickId, StartupInfo.ProcessOperationId, p.Path, 0, p.Plate, System.DBNull.Value, System.DBNull.Value);
                                preds.Add(p);
                            }
                        }
                        else if (ProgSettings.Source == PathSource.CSDoublet || ProgSettings.Source == PathSource.CSDoubletConnect)
                        {
                            SySal.BasicTypes.Vector BrickZero = new SySal.BasicTypes.Vector();
                            System.Data.DataSet dsbz = new System.Data.DataSet();
                            new SySal.OperaDb.OperaDbDataAdapter("SELECT ZEROX, ZEROY, ZEROZ FROM TB_EVENTBRICKS WHERE ID = " + StartupInfo.BrickId, Conn, null).Fill(dsbz);
                            BrickZero.X = SySal.OperaDb.Convert.ToDouble(dsbz.Tables[0].Rows[0][0]);
                            BrickZero.Y = SySal.OperaDb.Convert.ToDouble(dsbz.Tables[0].Rows[0][1]);
                            BrickZero.Z = SySal.OperaDb.Convert.ToDouble(dsbz.Tables[0].Rows[0][2]);

                            int pathid = 0;
                            double ZTarget;
                            ZTarget = (ProgSettings.Direction == ScanDirection.Upstream) ? MaxZ : MinZ;

                            csdoubletselcands = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("select /*+index_asc (tb_b_cscands_sbpaths pk_b_cscands_sbpaths) */ count(*) from tb_b_cscands_sbpaths where id_eventbrick = " + StartupInfo.BrickId + " and id_scanback_procopid is null ", Conn, null).ExecuteScalar());

                            System.Data.DataSet dps = new System.Data.DataSet();
                            new SySal.OperaDb.OperaDbDataAdapter(
                                "select id_cs_eventbrick, id_candidate, posx + zerox, posy + zeroy, slopex, slopey, id_plate, cs_side, Z + zeroz, track# from " +
                                "(" +
                                "select /*+index(tb_plates pk_plates) */ id_cs_eventbrick, id_candidate, posx, posy, slopex, slopey, id as id_plate, cs_side, decode(cs_side, 1, upz, 2, downz) as Z, row_number() over (partition by id_cs_eventbrick, id_candidate order by " + ((ProgSettings.Direction == ScanDirection.Upstream) ? "id_plate, cs_side" : "id_plate desc, cs_side desc") + ") as track# from " +
                                " (" +
                                "  select /*+index(tb_zones pk_zones) */ id_cs_eventbrick, id_candidate, posx, posy, slopex, slopey, id_cs_zone, cs_side, downz, upz, id_plate from " +
                                "  (" +
                                "   select /*+index(tb_views pk_views) */ id_cs_eventbrick, id_candidate, m_posx as posx, m_posy as posy, slopex, slopey, id_cs_zone, cs_side, downz, upz from " +
                                "   (" +
                                "    select id_cs_eventbrick, id_candidate, posx as m_posx, posy as m_posy, slopex, slopey, id_cs_zone, cs_side, id_view from " +
                                "    (" +
                                "     select /*+index_asc (tb_cs_candidate_tracks pk_cs_candidate_tracks) */ tb_cs_candidate_tracks.id_eventbrick as id_cs_eventbrick, id_candidate, id_zone as id_cs_zone, side as cs_side, id_microtrack from tb_cs_candidate_tracks where (id_eventbrick, id_candidate) in " +
                                "     (" +
                                (csdoubletselcands > 0 ? ("      select /*+index_asc (tb_b_cscands_sbpaths pk_b_cscands_sbpaths) */ id_cs_eventbrick, id_candidate from tb_b_cscands_sbpaths where id_eventbrick = " + StartupInfo.BrickId + " and id_scanback_procopid is null ") : ProgSettings.CSDoubletAutoSel.Replace("_BRICK_", StartupInfo.BrickId.ToString())) +
                                "     )" +
                                "    )" +
                                "    inner join tb_mipmicrotracks on (id_cs_eventbrick = id_eventbrick and id_zone = id_cs_zone and side = cs_side and id = id_microtrack) " +
                                "   )" +
                                "   inner join tb_views on (id_cs_eventbrick = id_eventbrick and id_zone = id_cs_zone and side = cs_side and id = id_view) " +
                                "  )" +
                                "  inner join tb_zones on (id_eventbrick = id_cs_eventbrick and id = id_cs_zone) " +
                                " )" +
                                " inner join tb_plates on (id_eventbrick = id_cs_eventbrick and id = id_plate)" +
                                ") inner join tb_eventbricks on (id_cs_eventbrick = id)" +
                                "order by id_cs_eventbrick, id_candidate, track# desc",
                                Conn, null).Fill(dps);

                            int predbase = 0;
                            System.Data.DataRowCollection drc = dps.Tables[0].Rows;
                            while (predbase < drc.Count)
                            {
                                System.Data.DataRow dr = drc[predbase];
                                int predmutks = SySal.OperaDb.Convert.ToInt32(dr[9]);
                                double lastposx = SySal.OperaDb.Convert.ToInt32(dr[2]);
                                double lastposy = SySal.OperaDb.Convert.ToInt32(dr[3]);
                                double lastz = SySal.OperaDb.Convert.ToInt32(dr[8]);
                                double slopex = 0.0, slopey = 0.0;
                                bool basetrackformed = false;
                                if (predmutks > 1)
                                {
                                    int platescan;
                                    for (platescan = predbase; platescan < (predbase + predmutks - 1); platescan++)
                                        if (SySal.OperaDb.Convert.ToInt32(drc[platescan][6]) == SySal.OperaDb.Convert.ToInt32(drc[platescan + 1][6]) && SySal.OperaDb.Convert.ToInt32(drc[platescan][7]) != SySal.OperaDb.Convert.ToInt32(drc[platescan + 1][7]))
                                        {
                                            double x1, y1, z1, x2, y2, z2;
                                            x1 = SySal.OperaDb.Convert.ToDouble(drc[platescan][2]);
                                            y1 = SySal.OperaDb.Convert.ToDouble(drc[platescan][3]);
                                            z1 = SySal.OperaDb.Convert.ToDouble(drc[platescan][8]);
                                            x2 = SySal.OperaDb.Convert.ToDouble(drc[platescan + 1][2]);
                                            y2 = SySal.OperaDb.Convert.ToDouble(drc[platescan + 1][3]);
                                            z2 = SySal.OperaDb.Convert.ToDouble(drc[platescan + 1][8]);
                                            slopex = (x1 - x2) / (z1 - z2);
                                            slopey = (y1 - y2) / (z1 - z2);
                                            basetrackformed = true;
                                            break;
                                        }
                                }
                                if (basetrackformed == false)
                                {
                                    int platescan;
                                    slopex = slopey = 0.0;
                                    for (platescan = predbase; platescan < (predbase + predmutks); platescan++)
                                    {
                                        slopex += SySal.OperaDb.Convert.ToDouble(drc[platescan][4]);
                                        slopey += SySal.OperaDb.Convert.ToDouble(drc[platescan][5]);
                                    }
                                    slopex /= predmutks;
                                    slopey /= predmutks;
                                }
                                Prediction p = new Prediction();
                                p.Path = ++pathid;
                                if (ProgSettings.CSDoubletDZOverride)
                                {
                                    p.Pos.X = lastposx + slopex * (ProgSettings.CSDoubletDZ - ((Plate == 1) ? 300 : 0)) - BrickZero.X;
                                    p.Pos.Y = lastposy + slopey * (ProgSettings.CSDoubletDZ - ((Plate == 1) ? 300 : 0)) - BrickZero.Y;
                                }
                                else
                                {
                                    p.Pos.X = lastposx + slopex * (ZTarget - lastz) - BrickZero.X;
                                    p.Pos.Y = lastposy + slopey * (ZTarget - lastz) - BrickZero.Y;
                                }
                                p.Slope.X = slopex;
                                p.Slope.Y = slopey;
                                p.IdCSEventBrick = SySal.OperaDb.Convert.ToInt64(dr[0]);
                                p.IdCandidate = SySal.OperaDb.Convert.ToInt64(dr[1]);
                                p.Plate = Plate;
                                p.IdPath = SySal.OperaDb.Schema.TB_SCANBACK_PATHS.Insert(StartupInfo.BrickId, StartupInfo.ProcessOperationId, p.Path, 0, p.Plate, System.DBNull.Value, System.DBNull.Value);
                                if (ProgSettings.Source == PathSource.CSDoubletConnect)
                                    SySal.OperaDb.Schema.TB_B_CSCANDS_SBPATHS.Insert(p.IdCSEventBrick, p.IdCandidate, StartupInfo.BrickId, StartupInfo.ProcessOperationId, p.Path);
                                preds.Add(p);
                                predbase += predmutks;
                            }
                            if (ProgSettings.Source == PathSource.CSDoubletConnect)
                                SySal.OperaDb.Schema.TB_B_CSCANDS_SBPATHS.Flush();
                        }
                    }
                    else
                        lock (Conn)
                        {
                            if (ProgSettings.PlatesToScan <= 0 ||
                                SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT COUNT(DISTINCT ID_PLATE) FROM TB_SCANBACK_PREDICTIONS WHERE (ID_EVENTBRICK, ID_PATH) IN (SELECT ID_EVENTBRICK, ID FROM TB_SCANBACK_PATHS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PROCESSOPERATION = " + StartupInfo.ProcessOperationId + ")", Conn).ExecuteScalar()) < ProgSettings.PlatesToScan)
                            {
                                int lastvalidplate = 0;
                                int seenonce = 0, neverseen = 0;
                                //double newZ = SySal.OperaDb.Convert.ToDouble(new SySal.OperaDb.OperaDbCommand("SELECT Z FROM VW_PLATES WHERE (ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + Plate + ")", Conn, null).ExecuteScalar());
                                double newZ = SySal.OperaDb.Convert.ToDouble(new SySal.OperaDb.OperaDbCommand("SELECT /*+index(tb_plates pk_plates) */ Z FROM TB_PLATES WHERE (ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + Plate + ")", Conn, null).ExecuteScalar());
                                System.Data.DataSet dps = new System.Data.DataSet();

                                string qs =
                                    "select idpb, id_path, Z, ppx, ppy, psx, psy, nvl2(grains, 1, 0) as found, nvl(grains, 0) as grains, nvl(areasum, 0) as areasum, nvl(sigma, 0) as sigma, nvl(posx, 0) as fpx, nvl(posy, 0) as fpy, nvl(slopex, 0) as fsx, nvl(slopey, 0) as fsy from " +
                                    "(select id_eventbrick as idpb, id_path, Z, id_zone as idz, id_candidate, ppx, ppy, psx, psy from " +
                                    "(select id_eventbrick, id_path, id_plate, id_zone, id_candidate, posx as ppx, posy as ppy, slopex as psx, slopey as psy from tb_scanback_predictions where (id_eventbrick, id_path) in " +
                                    "(select id_eventbrick, id_path from " +
                                    "(select id_eventbrick, id_path, sum(nvl2(id_candidate, 0, 1)) as nfound from tb_scanback_predictions where (id_eventbrick, id_path, id_plate) in " +
                                    "(select id_eventbrick, id as idpath, idp from tb_scanback_paths, " +
                                    "(select idb, idp, pnum, Z from " +
                                    "( " +
                                    ((ProgSettings.Direction == SimpleScanback3Driver.ScanDirection.Upstream) ?
                                    " select /*+index_asc(tb_plates pk_plates) */ id_eventbrick as idb, id as idp, Z, row_number() over (order by Z asc) as pnum from vw_plates where id_eventbrick = " + StartupInfo.BrickId + " and Z > (select Z from vw_plates where id_eventbrick = " + StartupInfo.BrickId + " and id = " + Plate + ") and damaged = 'N' " :
                                    " select /*+index_asc(tb_plates pk_plates) */ id_eventbrick as idb, id as idp, Z, row_number() over (order by Z desc) as pnum from vw_plates where id_eventbrick = " + StartupInfo.BrickId + " and Z < (select Z from vw_plates where id_eventbrick = " + StartupInfo.BrickId + " and id = " + Plate + ") and damaged = 'N' ") +
                                    ")" +
                                    "where pnum <= " + ProgSettings.MaxMissingPlates + ") " +
                                    "where id_eventbrick = " + StartupInfo.BrickId + " and id_processoperation = " + StartupInfo.ProcessOperationId + (ProgSettings.DoNotPropagateForkedPaths ? " and ID_FORK_PATH IS NULL " : "") + ") " +
                                    "group by id_eventbrick, id_path) where nfound < " + ProgSettings.MaxMissingPlates + ")) inner join (select idp, Z from " +
                                    "( " +
                                    //								" select /*+index_asc(tb_plates pk_plates) */ id_eventbrick as idb, id as idp, Z, row_number() over (order by Z asc) as pnum from vw_plates where id_eventbrick = " + StartupInfo.BrickId + " and Z > (select Z from vw_plates where id_eventbrick = " + StartupInfo.BrickId + " and id = " + Plate + ") and damaged = 'N' " :
                                    //								" select /*+index_asc(tb_plates pk_plates) */ id_eventbrick as idb, id as idp, Z, row_number() over (order by Z desc) as pnum from vw_plates where id_eventbrick = " + StartupInfo.BrickId + " and Z < (select Z from vw_plates where id_eventbrick = " + StartupInfo.BrickId + " and id = " + Plate + ") and damaged = 'N' ") + 								 
                                    //								"where pnum = 1) on (idp = id_plate)) " +
                                    " select /*+index(tb_plates pk_plates) */ id_eventbrick as idb, id as idp, Z from tb_plates where (id_eventbrick, id) in ( select idb, idp from (" +
                                    ((ProgSettings.Direction == SimpleScanback3Driver.ScanDirection.Upstream) ?
                                    " select /*+index_asc(tb_plates pk_plates) */ id_eventbrick as idb, id as idp, row_number() over (order by Z asc) as pnum from vw_plates where id_eventbrick = " + StartupInfo.BrickId + " and Z > (select Z from vw_plates where id_eventbrick = " + StartupInfo.BrickId + " and id = " + Plate + ") and damaged = 'N' " :
                                    " select /*+index_asc(tb_plates pk_plates) */ id_eventbrick as idb, id as idp, row_number() over (order by Z desc) as pnum from vw_plates where id_eventbrick = " + StartupInfo.BrickId + " and Z < (select Z from vw_plates where id_eventbrick = " + StartupInfo.BrickId + " and id = " + Plate + ") and damaged = 'N' ") +
                                    ") where pnum = 1))) on (idp = id_plate)) " +
                                    "left join tb_mipbasetracks on (id_eventbrick = idpb and id_zone = idz and id_candidate = id)";
                                try
                                {
                                    new SySal.OperaDb.OperaDbDataAdapter(qs, Conn, null).Fill(dps);
                                }
                                catch (Exception x)
                                {
                                    HE.WriteLine(x.ToString());
                                    throw x;
                                }
                                System.Data.DataRowCollection drc = dps.Tables[0].Rows;
                                foreach (System.Data.DataRow dr in drc)
                                {
                                    Prediction p = new Prediction();
                                    p.Plate = Plate;
                                    p.IdPath = Convert.ToInt64(dr[1]);
                                    seenonce++;

                                    int iter, deltac = 0;
                                    NumericalTools.CStyleParsedFunction f = null;
                                    for (iter = 0; iter < 4; iter++)
                                    {
                                        switch (iter)
                                        {
                                            case 0: f = PositionPropagationF; deltac = 0; break;
                                            case 1: f = PositionPropagationF; deltac = 1; break;
                                            case 2: f = SlopePropagationF; deltac = 0; break;
                                            case 3: f = SlopePropagationF; deltac = 1; break;
                                        }
                                        for (i = 0; i < f.ParameterList.Length; i++)
                                        {
                                            string s = f.ParameterList[i];
                                            if (String.Compare(s, "LP", true) == 0) f[i] = Convert.ToDouble(dr[3 + deltac]);
                                            else if (String.Compare(s, "LS", true) == 0) f[i] = Convert.ToDouble(dr[5 + deltac]);
                                            else if (String.Compare(s, "LZ", true) == 0) f[i] = Convert.ToDouble(dr[2]);
                                            else if (String.Compare(s, "FP", true) == 0) f[i] = Convert.ToDouble(dr[11 + deltac]);
                                            else if (String.Compare(s, "FS", true) == 0) f[i] = Convert.ToDouble(dr[13 + deltac]);
                                            else if (String.Compare(s, "Z", true) == 0) f[i] = newZ;
                                            else if (String.Compare(s, "F", true) == 0) f[i] = Convert.ToDouble(dr[7]);
                                            else if (String.Compare(s, "N", true) == 0) f[i] = Convert.ToDouble(dr[8]);
                                            else if (String.Compare(s, "A", true) == 0) f[i] = Convert.ToDouble(dr[9]);
                                            else if (String.Compare(s, "S", true) == 0) f[i] = Convert.ToDouble(dr[10]);
                                        }
                                        switch (iter)
                                        {
                                            case 0: p.Pos.X = f.Evaluate(); break;
                                            case 1: p.Pos.Y = f.Evaluate(); break;
                                            case 2: p.Slope.X = f.Evaluate(); break;
                                            case 3: p.Slope.Y = f.Evaluate(); break;
                                        }
                                    }

                                    preds.Add(p);
                                }
                                HE.WriteLine("Plate #" + Plate + " - Never seen: " + neverseen + " Seen once: " + seenonce + ".");
                            }
                        }
                    lock (Conn)
                    {
                        if (preds.Count == 0)
                        {
                            NoMorePredictions = true;
                            if (ProgSettings.Direction == ScanDirection.Upstream)
                            {
                                NextPredictedPlate = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("select max(idp) from (SELECT IDP FROM (SELECT IDP, ROW_NUMBER() OVER (ORDER BY Z DESC) as RNUM FROM " +
                                    "(SELECT ID_EVENTBRICK as IDB, ID as IDP, Z FROM TB_PLATES WHERE Z < (SELECT Z FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + Plate + ") AND ID_EVENTBRICK = " + StartupInfo.BrickId + ")" +
                                    "INNER JOIN (SELECT ID_EVENTBRICK, ID_PLATE, ID_PATH FROM TB_SCANBACK_PREDICTIONS WHERE (ID_EVENTBRICK, ID_PATH) IN (SELECT ID_EVENTBRICK, ID FROM TB_SCANBACK_PATHS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PROCESSOPERATION = " + StartupInfo.ProcessOperationId + "))" +
                                    "ON (ID_EVENTBRICK = IDB AND ID_PLATE = IDP)) WHERE RNUM = 1 UNION SELECT -1 AS IDP FROM DUAL)", Conn).ExecuteScalar());
                            }
                            else
                            {
                                NextPredictedPlate = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("select max(idp) from (SELECT IDP FROM (SELECT IDP, ROW_NUMBER() OVER (ORDER BY Z ASC) as RNUM FROM " +
                                    "(SELECT ID_EVENTBRICK as IDB, ID as IDP, Z FROM TB_PLATES WHERE Z > (SELECT Z FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + Plate + ") AND ID_EVENTBRICK = " + StartupInfo.BrickId + ")" +
                                    "INNER JOIN (SELECT ID_EVENTBRICK, ID_PLATE, ID_PATH FROM TB_SCANBACK_PREDICTIONS WHERE (ID_EVENTBRICK, ID_PATH) IN (SELECT ID_EVENTBRICK, ID FROM TB_SCANBACK_PATHS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PROCESSOPERATION = " + StartupInfo.ProcessOperationId + "))" +
                                    "ON (ID_EVENTBRICK = IDB AND ID_PLATE = IDP)) WHERE RNUM = 1 UNION SELECT -1 AS IDP FROM DUAL)", Conn).ExecuteScalar());
                            }
                        }
                        else
                        {
                            foreach (Prediction p in preds)
                                SySal.OperaDb.Schema.TB_SCANBACK_PREDICTIONS.Insert(StartupInfo.BrickId, p.IdPath, p.Plate, p.Pos.X, p.Pos.Y, p.Slope.X, p.Slope.Y, System.DBNull.Value, System.DBNull.Value, System.DBNull.Value, System.DBNull.Value, System.DBNull.Value, System.DBNull.Value, System.DBNull.Value, System.DBNull.Value, System.DBNull.Value);
                            SySal.OperaDb.Schema.TB_SCANBACK_PREDICTIONS.Flush();
                        }

                        if (isbrickentrypoint)
                        {
                            if (ProgSettings.Source == PathSource.CSDoublet && csdoubletselcands > 0)
                            {
                                long[] a_idcseventbrick = new long[BatchSize];
                                long[] a_idcscandidate = new long[BatchSize];
                                long[] a_path = new long[BatchSize];
                                SySal.OperaDb.OperaDbCommand cssbcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SET_CSCAND_SBPATH(:idcsbrick, :idcand, " + StartupInfo.BrickId + ", " + StartupInfo.ProcessOperationId + ", :path)", Conn, trans);
                                cssbcmd.Parameters.Add("idcsbrick", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
                                cssbcmd.Parameters.Add("idcand", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
                                cssbcmd.Parameters.Add("path", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
                                cssbcmd.ArrayBindCount = BatchSize;
                                cssbcmd.Prepare();
                                cssbcmd.Parameters[0].Value = a_idcseventbrick;
                                cssbcmd.Parameters[1].Value = a_idcscandidate;
                                cssbcmd.Parameters[2].Value = a_path;
                                for (i = 0; i < preds.Count; i += b)
                                {
                                    cssbcmd.ArrayBindCount = ((i + BatchSize) < preds.Count) ? BatchSize : (preds.Count - i);
                                    for (b = 0; b < BatchSize && (i + b) < preds.Count; b++)
                                    {
                                        Prediction p = (Prediction)preds[i + b];
                                        a_idcseventbrick[b] = p.IdCSEventBrick;
                                        a_idcscandidate[b] = p.IdCandidate;
                                        a_path[b] = p.Path;
                                        HE.WriteLine("Added CS candidate link for " + p.Path.ToString());
                                    }
                                    if (cssbcmd.ArrayBindCount > 0) cssbcmd.ExecuteNonQuery();
                                }
                            }
                            else if (ProgSettings.Source == PathSource.Prediction)
                            {
                                long[] a_idevent = new long[BatchSize];
                                long[] a_track = new long[BatchSize];
                                long[] a_path = new long[BatchSize];
                                SySal.OperaDb.OperaDbCommand prdcmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SET_PREDTRACK_SBPATH(:idev, :trk, " + StartupInfo.BrickId + ", " + StartupInfo.ProcessOperationId + ", :path)", Conn, trans);
                                prdcmd.Parameters.Add("idev", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
                                prdcmd.Parameters.Add("trk", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
                                prdcmd.Parameters.Add("path", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
                                prdcmd.ArrayBindCount = BatchSize;
                                prdcmd.Prepare();
                                prdcmd.Parameters[0].Value = a_idevent;
                                prdcmd.Parameters[1].Value = a_track;
                                prdcmd.Parameters[2].Value = a_path;
                                for (i = 0; i < preds.Count; i += b)
                                {
                                    prdcmd.ArrayBindCount = ((i + BatchSize) < preds.Count) ? BatchSize : (preds.Count - i);
                                    for (b = 0; b < BatchSize && (i + b) < preds.Count; b++)
                                    {
                                        Prediction p = (Prediction)preds[i + b];
                                        a_idevent[b] = p.IdEvent;
                                        a_track[b] = p.Track;
                                        a_path[b] = p.Path;
                                        HE.WriteLine("Added prediction link for " + p.Path.ToString());
                                    }
                                    if (prdcmd.ArrayBindCount > 0) prdcmd.ExecuteNonQuery();
                                }
                            }
                            NoMorePredictions = true;
                            if (ProgSettings.Direction == ScanDirection.Upstream)
                            {
                                NextPredictedPlate = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("select max(idp) from (SELECT IDP FROM (SELECT IDP, ROW_NUMBER() OVER (ORDER BY Z DESC) as RNUM FROM " +
                                    "(SELECT ID_EVENTBRICK as IDB, ID as IDP, Z FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + ")" +
                                    "INNER JOIN (SELECT ID_EVENTBRICK, ID_PLATE, ID_PATH FROM TB_SCANBACK_PREDICTIONS WHERE (ID_EVENTBRICK, ID_PATH) IN (SELECT ID_EVENTBRICK, ID FROM TB_SCANBACK_PATHS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PROCESSOPERATION = " + StartupInfo.ProcessOperationId + "))" +
                                    "ON (ID_EVENTBRICK = IDB AND ID_PLATE = IDP)) WHERE RNUM = " + (1 + ProgSettings.SkipPlates) + " UNION SELECT -1 AS IDP FROM DUAL)", Conn).ExecuteScalar());
                            }
                            else
                            {
                                NextPredictedPlate = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("select max(idp) from (SELECT IDP FROM (SELECT IDP, ROW_NUMBER() OVER (ORDER BY Z ASC) as RNUM FROM " +
                                    "(SELECT ID_EVENTBRICK as IDB, ID as IDP, Z FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + ")" +
                                    "INNER JOIN (SELECT ID_EVENTBRICK, ID_PLATE, ID_PATH FROM TB_SCANBACK_PREDICTIONS WHERE (ID_EVENTBRICK, ID_PATH) IN (SELECT ID_EVENTBRICK, ID FROM TB_SCANBACK_PATHS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PROCESSOPERATION = " + StartupInfo.ProcessOperationId + "))" +
                                    "ON (ID_EVENTBRICK = IDB AND ID_PLATE = IDP)) WHERE RNUM = " + (1 + ProgSettings.SkipPlates) + " UNION SELECT -1 AS IDP FROM DUAL)", Conn).ExecuteScalar());
                            }
                        }
                    }
                    HE.WriteLine("Plate #" + Plate + " - Predictions: " + preds.Count + ".");
                    if (preds.Count == 0) NoMorePredictions = true;

                    trans.Commit();
                    PredictionsDone = true;
                    UpdateProgress();
                    UpdatePlots();
                }
                HE.WriteLine("Plate #" + Plate + " Predictions OK");
                PredException = null;
            }
            catch (Exception x)
            {
                if (trans != null) trans.Rollback();
                PredException = new Exception("Error making predictions!\r\n" + x.Message);
                HE.WriteLine("Error making predictions!\r\n" + x.ToString());
            }
            finally
            {
                Conn.Close();
            }
            PredictionsDoneEvent.Set();
        }

        private static void Execute()
        {
            ThisThread = System.Threading.Thread.CurrentThread;
            gIm = new System.Drawing.Bitmap(500, 375);
            gMon = System.Drawing.Graphics.FromImage(gIm);
            gPlot = new NumericalTools.Plot();

            StartupInfo = (SySal.DAQSystem.Drivers.VolumeOperationInfo)HE.StartupInfo;
            Conn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
            Conn.Open();
            SySal.OperaDb.Schema.DB = Conn;

            System.Data.DataSet ds = new System.Data.DataSet();

            new SySal.OperaDb.OperaDbDataAdapter("with platesz as ( " +
                "  select /*+index_asc(tb_plates pk_plates) */ ID, Z from tb_plates where id_eventbrick = " + StartupInfo.BrickId + " order by z asc " +
                ") " +
                "select * from " +
                "(" +
                " select id as zid, z from platesz where z = (select min(z) from platesz) " +
                " union" +
                " select id as zid, z from platesz where z = (select max(z) from platesz) " +
                ") order by z", Conn, null).Fill(ds);
            MinPlate = Convert.ToInt32(ds.Tables[0].Rows[0][0]);
            MinZ = Convert.ToDouble(ds.Tables[0].Rows[0][1]);
            MaxPlate = Convert.ToInt32(ds.Tables[0].Rows[1][0]);
            MaxZ = Convert.ToDouble(ds.Tables[0].Rows[1][1]);

            System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(SimpleScanback3Settings));
            ProgSettings = (SimpleScanback3Settings)xmls.Deserialize(new System.IO.StringReader(HE.ProgramSettings));
            PositionPropagationF = new NumericalTools.CStyleParsedFunction(ProgSettings.PositionPropagationFunction);
            CheckVariables(PositionPropagationF);
            SlopePropagationF = new NumericalTools.CStyleParsedFunction(ProgSettings.SlopePropagationFunction);
            CheckVariables(SlopePropagationF);
            if (ProgSettings.UseToRecomputeCalibrationsOnly == true)
            {
                try
                {
                    ExecRecomputeCalibration();
                    ProgressInfo.Progress = 1.0;
                    ProgressInfo.Complete = true;
                    ProgressInfo.FinishTime = System.DateTime.Now;
                    UpdateProgress();
                    UpdatePlots();
                }
                finally
                {

                }
                return;
            }
            Conn.Close();
            if (ProgSettings.WaitForContinue) CanRun.Reset();
            if (ProgSettings.WaitForClose) CanEnd.Reset();
            if (ProgSettings.FirstPlatePredictionScanConfigId == 0) ProgSettings.FirstPlatePredictionScanConfigId = ProgSettings.PredictionScanConfigId;

            if (StartupInfo.ExeRepository.EndsWith("\\")) StartupInfo.ExeRepository = StartupInfo.ExeRepository.Remove(StartupInfo.ExeRepository.Length - 1, 1);
            if (StartupInfo.ScratchDir.EndsWith("\\")) StartupInfo.ScratchDir = StartupInfo.ScratchDir.Remove(StartupInfo.ScratchDir.Length - 1, 1);
            if (StartupInfo.LinkedZonePath.EndsWith("\\")) StartupInfo.LinkedZonePath = StartupInfo.LinkedZonePath.Remove(StartupInfo.LinkedZonePath.Length - 1, 1);
            if (StartupInfo.RawDataPath.EndsWith("\\")) StartupInfo.RawDataPath = StartupInfo.RawDataPath.Remove(StartupInfo.RawDataPath.Length - 1, 1);
            if (StartupInfo.RecoverFromProgressFile)
            {
                HE.WriteLine("Restarting from progress file");
                ProgressInfo = HE.ProgressInfo;
                try
                {
                    System.Xml.XmlDocument xmldoc = new System.Xml.XmlDocument();
                    xmldoc.LoadXml("<CustomInfo>" + ProgressInfo.CustomInfo.Replace('[', '<').Replace(']', '>') + "</CustomInfo>");
                    System.Xml.XmlNode xmln = xmldoc.FirstChild;
                    Plate = Convert.ToInt32(xmln["Plate"].InnerText);
                    WaitingOnId = Convert.ToInt64(xmln["WaitingOnId"].InnerText);
                    CalibrationId = Convert.ToInt64(xmln["CalibrationId"].InnerText);
                    IntercalibrationDone = Convert.ToBoolean(xmln["IntercalibrationDone"].InnerText);
                    PredictionsDone = Convert.ToBoolean(xmln["PredictionsDone"].InnerText);
                    ScanDone = Convert.ToBoolean(xmln["ScanDone"].InnerText);
                    System.Xml.XmlNode xmlin = xmln["IgnoreCalibrationList"];
                    if (xmlin != null)
                        foreach (System.Xml.XmlNode xmlln in xmlin.ChildNodes)
                        {
                            long ic = System.Convert.ToInt64(xmlln.InnerText);
                            int pos = IgnoreCalibrationList.BinarySearch(ic);
                            if (pos < 0) IgnoreCalibrationList.Insert(~pos, ic);
                        };
                    System.Xml.XmlNode xmlfn = xmln["ForceCalibrationList"];
                    if (xmlfn != null)
                        foreach (System.Xml.XmlNode xmlln in xmlfn.ChildNodes)
                        {
                            long ic = System.Convert.ToInt64(xmlln.InnerText);
                            int pos = ForceCalibrationList.BinarySearch(ic);
                            if (pos < 0) ForceCalibrationList.Insert(~pos, ic);
                        };
                    ProgressInfo.ExitException = null;
                    HE.WriteLine("Restarting complete");
                }
                catch (Exception)
                {
                    HE.WriteLine("Restarting failed - proceeding to re-initialize process.");
                    ProgressInfo = HE.ProgressInfo;
                    PredictionsDone = false;
                    IntercalibrationDone = false;
                    ScanDone = false;
                    WaitingOnId = 0;
                    Plate = (ProgSettings.Direction == ScanDirection.Upstream) ? MaxPlate : MinPlate;
                    ProgressInfo.Progress = 0.0;
                    ProgressInfo.StartTime = System.DateTime.Now;
                    ProgressInfo.FinishTime = ProgressInfo.StartTime.AddYears(1);
                }
            }
            else
            {
                ProgressInfo = HE.ProgressInfo;
                PredictionsDone = false;
                IntercalibrationDone = false;
                ScanDone = false;
                WaitingOnId = 0;
                Plate = (ProgSettings.Direction == ScanDirection.Upstream) ? MaxPlate : MinPlate;
                ProgressInfo.Progress = 0.0;
                ProgressInfo.StartTime = System.DateTime.Now;
                ProgressInfo.FinishTime = ProgressInfo.StartTime.AddYears(1);
            }
            UpdateProgress();
            UpdatePlots();

            PredictionsDoneEvent.Reset();
            WorkerThread.Start();
            Exe e = new Exe();
            HE.InterruptNotifier = e;
            AppDomain.CurrentDomain.SetData(SySal.DAQSystem.Drivers.HostEnv.WebAccessString, e);
            if (ProgSettings.Source == PathSource.VolumeTrack) throw new Exception("Source " + ProgSettings.Source.ToString() + " not implemented yet!");
            else if (/*ProgSettings.Source == PathSource.Interrupt &&*/
                    PredictionsDone == false &&
                    ((ProgSettings.Direction == ScanDirection.Downstream && Plate == MinPlate) ||
                    (ProgSettings.Direction == ScanDirection.Upstream && Plate == MaxPlate)))
            {
                WaitForPredictions = true;
                WorkQueue.Enqueue(Plate);
                WorkerThread.Interrupt();
                PredictionsDoneEvent.WaitOne();
                UpdateProgress();
                if (NoMorePredictions) Plate = NextPredictedPlate;
            }

            CanRun.WaitOne();
            HE.WriteLine("Entering Scanback cycle.");
            NoMorePredictions = false;
            while (
#if false			
			(ProgSettings.Direction == ScanDirection.Upstream && Plate <= MaxPlate) || (ProgSettings.Direction == ScanDirection.Downstream && Plate >= MinPlate))
#else
Plate >= 0
#endif
)
            {
                CanRun.WaitOne();
                SySal.DAQSystem.Drivers.Status status;
                ReloadStatus = false;
                if (ReloadStatus) continue;
                if (PredictionsDone == false)
                {
                    PredictionsDoneEvent.Reset();
                    lock (WorkQueue)
                    {
                        WorkQueue.Enqueue(Plate);
                        WorkerThread.Interrupt();
                    }
                }
                else PredictionsDoneEvent.Set();
                SySal.OperaDb.OperaDbConnection tempConn2 = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
                tempConn2.Open();
                bool check = String.Compare(new SySal.OperaDb.OperaDbCommand("SELECT DAMAGED FROM VW_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + Plate, tempConn2, null).ExecuteScalar().ToString(), "N", true) == 0;
                tempConn2.Close();
                if (check)
                {
                    // First step: Intercalibration
                    if (IntercalibrationDone == false && ProgSettings.IntercalibrationConfigId != ProgSettings.PredictionScanConfigId)
                    {
                        CalibrationId = 0;
                        if (WaitingOnId == 0)
                        {
                            object o = null;
                            SySal.OperaDb.OperaDbConnection tempConn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
                            tempConn.Open();
                            o = new SySal.OperaDb.OperaDbCommand("SELECT CALIBRATION FROM VW_PLATES WHERE (ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + Plate + ")", tempConn, null).ExecuteScalar();
                            if (ForceCalibrationList.Count > 0)
                            {
                                System.Data.DataSet dfc = new System.Data.DataSet();
                                string fcstr = "";
                                foreach (long ic in ForceCalibrationList)
                                    if (fcstr.Length > 0) fcstr += "," + ic.ToString();
                                    else fcstr = ic.ToString();
                                new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_PROCESSOPERATION FROM TB_PLATE_CALIBRATIONS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PLATE = " + Plate + " AND ID_PROCESSOPERATION IN (" + fcstr + ")", tempConn, null).Fill(dfc);
                                tempConn.Close();
                                if (dfc.Tables[0].Rows.Count > 1)
                                {
                                    fcstr = "";
                                    foreach (System.Data.DataRow dr in dfc.Tables[0].Rows)
                                        fcstr += "\r\n" + dr[0];
                                    throw new Exception("Ambiguity in calibration specification - found the following conflicting calibrations:" + fcstr);
                                }
                                else if (dfc.Tables[0].Rows.Count == 1) CalibrationId = SySal.OperaDb.Convert.ToInt64(dfc.Tables[0].Rows[0][0]);
                            }
                            else tempConn.Close();

                            if (CalibrationId == 0 && (o == System.DBNull.Value || o == null || (ProgSettings.ForceRefreshCalibration && SySal.OperaDb.Convert.ToInt64(o) < StartupInfo.ProcessOperationId) || (IgnoreCalibrationList.BinarySearch(SySal.OperaDb.Convert.ToInt64(o)) >= 0)))
                            {
                                SySal.DAQSystem.Drivers.ScanningStartupInfo intercalstartupinfo = new SySal.DAQSystem.Drivers.ScanningStartupInfo();
                                intercalstartupinfo.DBPassword = StartupInfo.DBPassword;
                                intercalstartupinfo.DBServers = StartupInfo.DBServers;
                                intercalstartupinfo.DBUserName = StartupInfo.DBUserName;
                                intercalstartupinfo.ExeRepository = StartupInfo.ExeRepository;
                                intercalstartupinfo.LinkedZonePath = StartupInfo.LinkedZonePath;
                                intercalstartupinfo.MachineId = StartupInfo.MachineId;
                                intercalstartupinfo.Plate = new SySal.DAQSystem.Scanning.MountPlateDesc();
                                intercalstartupinfo.Plate.BrickId = StartupInfo.BrickId;
                                intercalstartupinfo.Plate.PlateId = Plate;
                                intercalstartupinfo.Plate.MapInitString = "";
                                intercalstartupinfo.Plate.TextDesc = "Brick #" + intercalstartupinfo.Plate.BrickId + " Plate #" + intercalstartupinfo.Plate.PlateId;
                                intercalstartupinfo.ProcessOperationId = 0;
                                intercalstartupinfo.CalibrationId = 0;
                                intercalstartupinfo.ProgramSettingsId = ProgSettings.IntercalibrationConfigId;
                                intercalstartupinfo.ProgressFile = "";
                                intercalstartupinfo.RawDataPath = StartupInfo.RawDataPath;
                                intercalstartupinfo.RecoverFromProgressFile = false;
                                intercalstartupinfo.ScratchDir = StartupInfo.ScratchDir;
                                intercalstartupinfo.Zones = new SySal.DAQSystem.Scanning.ZoneDesc[0];
                                CalibrationId = WaitingOnId = HE.Start(intercalstartupinfo);
                                UpdateProgress();
                            }
                            else if (CalibrationId == 0 && o != null && o != System.DBNull.Value) CalibrationId = SySal.OperaDb.Convert.ToInt64(o);
                        }
                        if (WaitingOnId != 0)
                        {
                            status = HE.Wait(WaitingOnId);
                            lock (StartupInfo)
                                if (ReloadStatus)
                                    continue;
                            if (status == SySal.DAQSystem.Drivers.Status.Failed)
                            {
                                WaitingOnId = 0;
                                throw new Exception("Intercalibration failed on brick " + StartupInfo.BrickId + " , plate " + Plate + "!\r\nScanback interrupted.");
                            }
                        }
                        IntercalibrationDone = true;
                        WaitingOnId = 0;
                        UpdateProgress();
                        HE.WriteLine("Plate #" + Plate + " Intercalibration OK");
                    }
                    else HE.WriteLine("Plate #" + Plate + " Intercalibration " + (IntercalibrationDone ? "OK" : "skipped because of program settings."));
                    PredictionsDoneEvent.WaitOne();
                    UpdateProgress();
                    if (NoMorePredictions)
                    {
                        Plate = NextPredictedPlate;
                        NoMorePredictions = false;
                        continue;
                    }
                    // Third step: scanning
                    if (PredictionsDone == false)
                    {
                        if (PredException != null) throw PredException;
                        else throw new Exception("Execution inconsistency found! Prediction thread probably aborted!");
                    }
                    lock (StartupInfo)
                        if (ReloadStatus)
                            continue;
                    CanRun.WaitOne();
                    if (ScanDone == false)
                    {
                        if (WaitingOnId == 0)
                        {
                            //Conn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
                            Conn.Open();
                            SySal.DAQSystem.Drivers.ScanningStartupInfo predscanstartupinfo = new SySal.DAQSystem.Drivers.ScanningStartupInfo();
                            predscanstartupinfo.DBPassword = StartupInfo.DBPassword;
                            predscanstartupinfo.DBServers = StartupInfo.DBServers;
                            predscanstartupinfo.DBUserName = StartupInfo.DBUserName;
                            predscanstartupinfo.ExeRepository = StartupInfo.ExeRepository;
                            predscanstartupinfo.LinkedZonePath = StartupInfo.LinkedZonePath;
                            predscanstartupinfo.MachineId = StartupInfo.MachineId;
                            predscanstartupinfo.Plate = new SySal.DAQSystem.Scanning.MountPlateDesc();
                            predscanstartupinfo.Plate.BrickId = StartupInfo.BrickId;
                            predscanstartupinfo.Plate.PlateId = Plate;
                            predscanstartupinfo.Plate.TextDesc = "Brick #" + predscanstartupinfo.Plate.BrickId + " Plate #" + predscanstartupinfo.Plate.PlateId;
                            predscanstartupinfo.ProcessOperationId = 0;
                            predscanstartupinfo.ProgramSettingsId = ((ProgSettings.Direction == ScanDirection.Upstream && Plate == MaxPlate) || (ProgSettings.Direction == ScanDirection.Downstream && Plate == MinPlate)) ? ProgSettings.FirstPlatePredictionScanConfigId : ProgSettings.PredictionScanConfigId;
                            predscanstartupinfo.Plate.MapInitString = (CalibrationId > 0)
                                ? SySal.OperaDb.Scanning.Utilities.GetMapString(predscanstartupinfo.Plate.BrickId, predscanstartupinfo.Plate.PlateId, predscanstartupinfo.CalibrationId = CalibrationId, Conn, null)
                                : SySal.OperaDb.Scanning.Utilities.GetMapString(predscanstartupinfo.Plate.BrickId, predscanstartupinfo.Plate.PlateId, false, SySal.OperaDb.Scanning.Utilities.CharToMarkType(Convert.ToChar(new SySal.OperaDb.OperaDbCommand("SELECT MARKSET FROM TB_PROGRAMSETTINGS WHERE ID = " + predscanstartupinfo.ProgramSettingsId, Conn).ExecuteScalar().ToString().Trim())),
                                    out predscanstartupinfo.CalibrationId, Conn, null);
                            predscanstartupinfo.ProgressFile = "";
                            predscanstartupinfo.RawDataPath = StartupInfo.RawDataPath;
                            predscanstartupinfo.RecoverFromProgressFile = false;
                            predscanstartupinfo.ScratchDir = StartupInfo.ScratchDir;
                            predscanstartupinfo.Zones = new SySal.DAQSystem.Scanning.ZoneDesc[0];
                            WaitingOnId = HE.Start(predscanstartupinfo);
                            UpdateProgress();
                            UpdatePlots();
                            Conn.Close();
                        }
                        status = HE.Wait(WaitingOnId);
                        lock (StartupInfo)
                            if (ReloadStatus)
                                continue;
                        if (status == SySal.DAQSystem.Drivers.Status.Failed)
                        {
                            WaitingOnId = 0;
                            throw new Exception("Scanning failed on brick " + StartupInfo.BrickId + " , plate " + Plate + "!\r\nScanback interrupted.");
                        }
                        ScanDone = true;
                        if (ProgSettings.IntercalibrationConfigId == ProgSettings.PredictionScanConfigId) IntercalibrationDone = true;
                        WaitingOnId = 0;
                        UpdateProgress();
                        UpdatePlots();
                    }
                    HE.WriteLine("Plate #" + Plate + " Scanning OK");
                }
                else
                {
                    HE.WriteLine("Plate #" + Plate + " scanning skipped because of plate damage condition; deleting predictions for this plate.");
                    //Conn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
                    Conn.Open();
                    new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_DELETE_PREDICTIONS(" + StartupInfo.BrickId + ", " + StartupInfo.ProcessOperationId + ", " + Plate + ")", Conn, null).ExecuteNonQuery();
                    Conn.Close();
                    HE.WriteLine("Plate #" + Plate + " Skipped, moving to next plate.");
                }

                if (ProgSettings.Direction == ScanDirection.Upstream)
                {
                    //Conn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
                    Conn.Open();
                    object o = new SySal.OperaDb.OperaDbCommand("select id from (select /*+index_asc(tb_plates pk_plates) */ ID, row_number() over (order by z desc) as rnum from (select idp as id, zp as z from (select id_eventbrick as idb, id as idp, z as zp from tb_plates where id_eventbrick = "
                        + StartupInfo.BrickId + " and z < (select /*+index(tb_plates pk_plates) */ z from tb_plates where id_eventbrick = " + StartupInfo.BrickId + " and id = " + Plate + ") and id <> " + Plate + ") inner join vw_plates on (id_eventbrick = idb and idp = id and damaged = 'N'))) where rnum = 1",
                        Conn, null).ExecuteScalar();
                    Conn.Close();
                    if (o == System.DBNull.Value || o == null) Plate = -1;
                    else Plate = SySal.OperaDb.Convert.ToInt32(o);
                }
                else
                {
                    //Conn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
                    Conn.Open();
                    object o = new SySal.OperaDb.OperaDbCommand("select id from (select /*+index_asc(tb_plates pk_plates) */ ID, row_number() over (order by z asc) as rnum from (select idp as id, zp as z from (select id_eventbrick as idb, id as idp, z as zp from tb_plates where id_eventbrick = "
                        + StartupInfo.BrickId + " and z > (select /*+index(tb_plates pk_plates) */ z from tb_plates where id_eventbrick = " + StartupInfo.BrickId + " and id = " + Plate + ") and id <> " + Plate + ") inner join vw_plates on (id_eventbrick = idb and idp = id and damaged = 'N'))) where rnum = 1",
                        Conn, null).ExecuteScalar();
                    Conn.Close();
                    if (o == System.DBNull.Value || o == null) Plate = -1;
                    else Plate = SySal.OperaDb.Convert.ToInt32(o);
                }
                PredictionsDone = false;
                IntercalibrationDone = false;
                ScanDone = false;
                ProgressInfo.FinishTime = ProgressInfo.StartTime + System.TimeSpan.FromMilliseconds((System.DateTime.Now - ProgressInfo.StartTime).TotalMilliseconds * (1.0 - ProgressInfo.Progress));
                UpdateProgress();
                UpdatePlots();
            }
            if (ProgSettings.WaitForClose) HE.WriteLine("Waiting for Close");
            CanEnd.WaitOne();
            lock (WorkQueue)
            {
                WorkQueue.Clear();
                WorkerThread.Interrupt();
            }
            WorkerThread.Join();
            ProgressInfo.Progress = 1.0;
            ProgressInfo.Complete = true;
            ProgressInfo.FinishTime = System.DateTime.Now;
            UpdateProgress();
            UpdatePlots();
        }

        private static System.Text.RegularExpressions.Regex ForceCalEx = new System.Text.RegularExpressions.Regex(@"(\d+)\s+");

        private static System.Text.RegularExpressions.Regex IgnoreCalEx = new System.Text.RegularExpressions.Regex(@"(\d+)\s+");

        private static System.Text.RegularExpressions.Regex PathsEx = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\d+)");

        private static System.Text.RegularExpressions.Regex SetPathsEx = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\d+)");

        private static System.Text.RegularExpressions.Regex r_sptok = new System.Text.RegularExpressions.Regex(@"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*");

        private static System.Text.RegularExpressions.Regex TwoEx = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\d+)\s*");

        private static System.Text.RegularExpressions.Regex r_tok = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*");

        #region IInterruptNotifier Members

        private static void GetZ(int newplate, int plate, out double newZ, out double currZ)
        {
            System.Data.DataSet ds = new System.Data.DataSet();
            //Conn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
            Conn.Open();
            SySal.OperaDb.OperaDbDataAdapter da = new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, Z, RECORDNAME FROM (SELECT ID, Z, 0 AS RECORDNAME FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + newplate + " UNION SELECT ID, Z, 1 AS RECORDNAME FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + plate + ") ORDER BY RECORDNAME", Conn, null);
            da.Fill(ds);
            Conn.Close();
            newZ = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[0][1]);
            currZ = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[1][1]);
        }

        /// <summary>
        /// Receives interrupt notifications and processes each one immediately.
        /// </summary>
        /// <param name="nextint">the new interrupt information received.</param>
        public void NotifyInterrupt(Interrupt nextint)
        {
            SySal.OperaDb.OperaDbConnection tempConn = null;
            lock (StartupInfo)
            {
                try
                {
                    tempConn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
                    tempConn.Open();
                    HE.WriteLine("Processing interrupt string:\n" + nextint.Data);
                    if (nextint.Data != null && nextint.Data.Length > 0)
                    {
                        string[] lines = nextint.Data.Split(',');
                        char PlateDamagedCode = 'N';
                        int PlateDamaged = Plate;
                        int GoBackToPlateN = -1;
                        bool CancelCalibrations = false;
                        bool AbortCurrent = false;

                        bool specPlateDamagedCode = false;
                        bool specGoBackToPlateN = false;

                        foreach (string line in lines)
                        {
                            if (String.Compare(line.Trim(), "Continue", true) == 0)
                            {
                                CanRun.Set();
                                HE.WriteLine("CanRun Set");
                                ReloadStatus = false;
                                continue;
                            }
                            if (String.Compare(line.Trim(), "Close", true) == 0)
                            {
                                CanEnd.Set();
                                HE.WriteLine("CanEnd Set");
                                ReloadStatus = false;
                                continue;
                            }
                            System.Text.RegularExpressions.Match m = TwoEx.Match(line);
                            if (m.Success)
                            {
                                if (String.Compare(m.Groups[1].Value, "SetPaths", true) == 0)
                                {
                                    HE.WriteLine("SetPath detected");
                                    SySal.OperaDb.OperaDbTransaction sptrans = null;
                                    try
                                    {
                                        HE.WriteLine("getting number");
                                        int spn = System.Convert.ToInt32(m.Groups[2].Value);
                                        HE.WriteLine(spn.ToString());
                                        string[] spstr = line.Split(';');
                                        if (spstr.Length != spn + 1) throw new Exception("SetPath count does not match actual number of path information rows.");
                                        int i;
                                        lock (WorkQueue)
                                        {
                                            System.Collections.Generic.Dictionary<int, long> pathmap = new System.Collections.Generic.Dictionary<int, long>();
                                            HE.WriteLine("pathmap prepared");
                                            SySal.OperaDb.OperaDbDataReader pthr = new SySal.OperaDb.OperaDbCommand("SELECT PATH, ID FROM TB_SCANBACK_PATHS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PROCESSOPERATION = " + StartupInfo.ProcessOperationId, tempConn).ExecuteReader();
                                            while (pthr.Read())
                                                pathmap.Add(SySal.OperaDb.Convert.ToInt32(pthr.GetInt32(0)), SySal.OperaDb.Convert.ToInt64(pthr.GetInt64(1)));
                                            HE.WriteLine("pathmap read");
                                            SySal.OperaDb.Schema.DB = tempConn;
                                            sptrans = tempConn.BeginTransaction();
                                            for (i = 1; i <= spn; i++)
                                            {
                                                m = r_sptok.Match(spstr[i]);
                                                HE.WriteLine("processing " + spstr[i]);
                                                if (m.Success == false) throw new Exception("Wrong syntax found at setpath row " + i + ".");
                                                long pathid = pathmap[Convert.ToInt32(m.Groups[1].Value)];
                                                long zoneid = Convert.ToInt64(m.Groups[2].Value);
                                                int idcand = Convert.ToInt32(m.Groups[3].Value);
                                                int downcand = Convert.ToInt32(m.Groups[4].Value);
                                                int upcand = Convert.ToInt32(m.Groups[5].Value);
                                                HE.Write(i + " " + pathid + " " + zoneid + " " + idcand + " " + downcand + " " + upcand);
                                                SySal.OperaDb.Schema.TB_ZONES tbz = SySal.OperaDb.Schema.TB_ZONES.SelectPrimaryKey(StartupInfo.BrickId, zoneid, SySal.OperaDb.Schema.OrderBy.None);
                                                tbz.Row = 0;
                                                int plate = (int)tbz._ID_PLATE;
                                                HE.WriteLine(" " + plate);
                                                if (idcand <= 0)
                                                {
                                                    HE.WriteLine("IDCAND <= 0");
                                                    if (downcand <= 0 && upcand <= 0)
                                                    {
                                                        HE.WriteLine("No candidate");
                                                        SySal.OperaDb.Schema.LZ_SCANBACK_NOCANDIDATE.Insert(StartupInfo.BrickId, plate, pathid, zoneid);
                                                        HE.WriteLine("Continue");
                                                        continue;
                                                    }
                                                    if (downcand <= 0)
                                                    {
                                                        HE.WriteLine("DOWNCAND <= 0");
                                                        downcand = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT NVL(MAX(ID), 0) FROM TB_MIPMICROTRACKS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_ZONE = " + zoneid + " AND SIDE = 1", tempConn, sptrans).ExecuteScalar()) + 1;
                                                        HE.WriteLine("downcand = " + downcand);
                                                        SySal.OperaDb.Schema.TB_MIPMICROTRACKS upmk = SySal.OperaDb.Schema.TB_MIPMICROTRACKS.SelectPrimaryKey(StartupInfo.BrickId, zoneid, 2, upcand, SySal.OperaDb.Schema.OrderBy.None);
                                                        upmk.Row = 0;
                                                        HE.WriteLine("upmk");
                                                        SySal.OperaDb.Schema.TB_VIEWS upvw = SySal.OperaDb.Schema.TB_VIEWS.SelectPrimaryKey(StartupInfo.BrickId, zoneid, 2, upmk._ID_VIEW, SySal.OperaDb.Schema.OrderBy.None);
                                                        upvw.Row = 0;
                                                        HE.WriteLine("upvw");
                                                        SySal.OperaDb.Schema.TB_VIEWS dwvw = SySal.OperaDb.Schema.TB_VIEWS.SelectPrimaryKey(StartupInfo.BrickId, zoneid, 1, upmk._ID_VIEW, SySal.OperaDb.Schema.OrderBy.None);
                                                        dwvw.Row = 0;
                                                        HE.WriteLine("dwvw");
                                                        SySal.OperaDb.Schema.LZ_MIPMICROTRACKS.Insert(StartupInfo.BrickId, zoneid, 1, downcand, upmk._POSX + (dwvw._UPZ - upvw._DOWNZ) * upmk._SLOPEX, upmk._POSY + (dwvw._UPZ - upvw._DOWNZ) * upmk._SLOPEY, upmk._SLOPEX, upmk._SLOPEY, upmk._GRAINS, upmk._AREASUM, upmk._PH, 0.0, upmk._ID_VIEW);
                                                        HE.WriteLine("Insert microtrack");
                                                        idcand = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT NVL(MAX(ID), 0) FROM TB_MIPBASETRACKS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_ZONE = " + zoneid, tempConn, sptrans).ExecuteScalar()) + 1;
                                                        HE.WriteLine("idcand = " + idcand);
                                                        SySal.OperaDb.Schema.LZ_MIPBASETRACKS.Insert(StartupInfo.BrickId, zoneid, idcand, upmk._POSX + (dwvw._UPZ - upvw._DOWNZ) * upmk._SLOPEX, upmk._POSY + (dwvw._UPZ - upvw._DOWNZ) * upmk._SLOPEY, upmk._SLOPEX, upmk._SLOPEY, upmk._GRAINS, upmk._AREASUM, upmk._PH, -1.0, 1, downcand, 2, upcand);
                                                        HE.WriteLine("Insert basetrack");
                                                    }
                                                    else if (upcand <= 0)
                                                    {
                                                        HE.WriteLine("UPCAND <= 0");
                                                        upcand = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT NVL(MAX(ID), 0) FROM TB_MIPMICROTRACKS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_ZONE = " + zoneid + " AND SIDE = 2", tempConn, sptrans).ExecuteScalar()) + 1;
                                                        HE.WriteLine("upcand = " + downcand);
                                                        SySal.OperaDb.Schema.TB_MIPMICROTRACKS dwmk = SySal.OperaDb.Schema.TB_MIPMICROTRACKS.SelectPrimaryKey(StartupInfo.BrickId, zoneid, 1, downcand, SySal.OperaDb.Schema.OrderBy.None);
                                                        dwmk.Row = 0;
                                                        HE.WriteLine("dwmk");
                                                        SySal.OperaDb.Schema.TB_VIEWS dwvw = SySal.OperaDb.Schema.TB_VIEWS.SelectPrimaryKey(StartupInfo.BrickId, zoneid, 1, dwmk._ID_VIEW, SySal.OperaDb.Schema.OrderBy.None);
                                                        dwvw.Row = 0;
                                                        HE.WriteLine("dwvw");
                                                        SySal.OperaDb.Schema.TB_VIEWS upvw = SySal.OperaDb.Schema.TB_VIEWS.SelectPrimaryKey(StartupInfo.BrickId, zoneid, 2, dwmk._ID_VIEW, SySal.OperaDb.Schema.OrderBy.None);
                                                        upvw.Row = 0;
                                                        HE.WriteLine("upvw");
                                                        SySal.OperaDb.Schema.LZ_MIPMICROTRACKS.Insert(StartupInfo.BrickId, zoneid, 2, upcand, dwmk._POSX, dwmk._POSY, dwmk._SLOPEX, dwmk._SLOPEY, dwmk._GRAINS, dwmk._AREASUM, dwmk._PH, 0.0, dwmk._ID_VIEW);
                                                        HE.WriteLine("Insert microtrack");
                                                        idcand = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT NVL(MAX(ID), 0) FROM TB_MIPBASETRACKS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_ZONE = " + zoneid, tempConn, sptrans).ExecuteScalar()) + 1;
                                                        HE.WriteLine("idcand = " + idcand);
                                                        SySal.OperaDb.Schema.LZ_MIPBASETRACKS.Insert(StartupInfo.BrickId, zoneid, idcand, dwmk._POSX, dwmk._POSY, dwmk._SLOPEX, dwmk._SLOPEY, dwmk._GRAINS, dwmk._AREASUM, dwmk._PH, -1.0, 1, downcand, 2, upcand);
                                                        HE.WriteLine("Insert basetrack");
                                                    }
                                                    else
                                                    {
                                                        HE.WriteLine("UPCAND,DOWNCAND > 0");
                                                        SySal.OperaDb.Schema.TB_MIPMICROTRACKS dwmk = SySal.OperaDb.Schema.TB_MIPMICROTRACKS.SelectPrimaryKey(StartupInfo.BrickId, zoneid, 1, downcand, SySal.OperaDb.Schema.OrderBy.None);
                                                        dwmk.Row = 0;
                                                        HE.WriteLine("dwmk");
                                                        SySal.OperaDb.Schema.TB_MIPMICROTRACKS upmk = SySal.OperaDb.Schema.TB_MIPMICROTRACKS.SelectPrimaryKey(StartupInfo.BrickId, zoneid, 2, upcand, SySal.OperaDb.Schema.OrderBy.None);
                                                        upmk.Row = 0;
                                                        HE.WriteLine("upmk");
                                                        SySal.OperaDb.Schema.TB_VIEWS dwvw = SySal.OperaDb.Schema.TB_VIEWS.SelectPrimaryKey(StartupInfo.BrickId, zoneid, 1, dwmk._ID_VIEW, SySal.OperaDb.Schema.OrderBy.None);
                                                        dwvw.Row = 0;
                                                        HE.WriteLine("dwvw");
                                                        SySal.OperaDb.Schema.TB_VIEWS upvw = SySal.OperaDb.Schema.TB_VIEWS.SelectPrimaryKey(StartupInfo.BrickId, zoneid, 2, upmk._ID_VIEW, SySal.OperaDb.Schema.OrderBy.None);
                                                        upvw.Row = 0;
                                                        HE.WriteLine("upvw");
                                                        idcand = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT NVL(MAX(ID), 0) FROM TB_MIPBASETRACKS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_ZONE = " + zoneid, tempConn, sptrans).ExecuteScalar()) + 1;
                                                        HE.WriteLine("idcand = " + idcand);
                                                        SySal.OperaDb.Schema.LZ_MIPBASETRACKS.Insert(StartupInfo.BrickId, zoneid, idcand, dwmk._POSX, dwmk._POSY, (dwmk._POSX - upmk._POSX) / (dwvw._UPZ - upvw._DOWNZ), (dwmk._POSY - upmk._POSY) / (dwvw._UPZ - upvw._DOWNZ), SySal.OperaDb.Convert.ToInt32(dwmk._GRAINS) + SySal.OperaDb.Convert.ToInt32(upmk._GRAINS), SySal.OperaDb.Convert.ToInt32(dwmk._AREASUM) + SySal.OperaDb.Convert.ToInt32(upmk._AREASUM), System.DBNull.Value, 0.0, 1, downcand, 2, upcand);
                                                        HE.WriteLine("Insert basetrack");
                                                    }
                                                }
                                                HE.WriteLine("About to insert candidate");
                                                SySal.OperaDb.Schema.LZ_SCANBACK_CANDIDATE.Insert(StartupInfo.BrickId, plate, pathid, zoneid, idcand, 0);
                                                HE.WriteLine("Inserted");
                                            }
                                            HE.WriteLine("Microtrack flush");
                                            SySal.OperaDb.Schema.LZ_MIPMICROTRACKS.Flush();
                                            HE.WriteLine("Basetrack flush");
                                            SySal.OperaDb.Schema.LZ_MIPBASETRACKS.Flush();
                                            HE.WriteLine("Candidate flush");
                                            SySal.OperaDb.Schema.LZ_SCANBACK_CANDIDATE.Flush();
                                            HE.WriteLine("Nocandidate flush");
                                            SySal.OperaDb.Schema.LZ_SCANBACK_NOCANDIDATE.Flush();
                                            HE.WriteLine("Inserted " + spn + " setpath rows, committing.");
                                            sptrans.Commit();
                                            HE.WriteLine("OK - Closing allowed.");
                                            CanEnd.Set();
                                        }
                                    }
                                    catch (Exception x)
                                    {
                                        HE.WriteLine("Exception setting paths: " + x.ToString());
                                        if (sptrans != null) sptrans.Rollback();
                                    }
                                }
                                else if (WaitForPredictions)
                                {
                                    m = PathsEx.Match(line);
                                    if (m.Success == false) continue;
                                    Prediction[] predsread = null;
                                    if (String.Compare(m.Groups[1].Value, "Paths", true) == 0)
                                    {
                                        int ExpectedPredictions = 0;
                                        try
                                        {
                                            ExpectedPredictions = Convert.ToInt32(m.Groups[2].Value);
                                            predsread = new Prediction[ExpectedPredictions];
                                            int pr = 0;
                                            string[] prstr = line.Split(';');
                                            if (prstr.Length != ExpectedPredictions + 1) throw new Exception("Prediction count does not match actual number of predictions.");
                                            for (pr = 0; pr < predsread.Length; pr++)
                                            {
                                                m = r_tok.Match(prstr[pr + 1]);
                                                if (m.Success != true || m.Length != prstr[pr + 1].Length) throw new Exception("Wrong prediction syntax at prediction " + pr);
                                                Prediction p = new Prediction();
                                                p.IdPath = Convert.ToInt32(m.Groups[1].Value);
                                                p.Pos.X = Convert.ToDouble(m.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                p.Pos.Y = Convert.ToDouble(m.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                p.Slope.X = Convert.ToDouble(m.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                p.Slope.Y = Convert.ToDouble(m.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                p.Plate = Convert.ToInt32(m.Groups[6].Value);
                                                predsread[pr] = p;
                                            }
                                            PredictionsReceived = predsread;
                                            WaitForPredictions = false;
                                            PredictionsReceivedEvent.Set();
                                            HE.WriteLine("Received " + predsread.Length + " prediction(s).");
                                        }
                                        catch (Exception)
                                        {
                                            PredictionsReceived = predsread = null;
                                            WaitForPredictions = true;
                                            PredictionsReceivedEvent.Reset();
                                            continue;
                                        }
                                    }
                                };
                                if (String.Compare(m.Groups[1].Value, "ForceCalibrations", true) == 0)
                                {
                                    try
                                    {
                                        int ExpectedCalibrations = System.Convert.ToInt32(m.Groups[2].Value);
                                        System.Text.RegularExpressions.MatchCollection ms = ForceCalEx.Matches(line + " ", m.Length);
                                        HE.WriteLine("Expected " + ExpectedCalibrations + " Found " + ms.Count);
                                        if (ExpectedCalibrations != ms.Count) throw new Exception("Number of expected calibrations does not match the number of IDs found.");
                                        foreach (System.Text.RegularExpressions.Match mt in ms)
                                        {
                                            long ic = System.Convert.ToInt64(mt.Groups[1].Value);
                                            int pos = ForceCalibrationList.BinarySearch(ic);
                                            if (pos < 0) ForceCalibrationList.Insert(~pos, ic);
                                        }
                                        UpdateProgress();
                                    }
                                    catch (Exception x)
                                    {
                                        HE.WriteLine(x.Message);
                                    }
                                }
                                else if (String.Compare(m.Groups[1].Value, "IgnoreCalibrations", true) == 0)
                                {
                                    try
                                    {
                                        int ExpectedCalibrations = System.Convert.ToInt32(m.Groups[2].Value);
                                        System.Text.RegularExpressions.MatchCollection ms = IgnoreCalEx.Matches(line + " ", m.Length);
                                        HE.WriteLine("Expected " + ExpectedCalibrations + " Found " + ms.Count);
                                        if (ExpectedCalibrations != ms.Count) throw new Exception("Number of expected calibrations does not match the number of IDs found.");
                                        foreach (System.Text.RegularExpressions.Match mt in ms)
                                        {
                                            long ic = System.Convert.ToInt64(mt.Groups[1].Value);
                                            int pos = IgnoreCalibrationList.BinarySearch(ic);
                                            if (pos < 0) IgnoreCalibrationList.Insert(~pos, ic);
                                        }
                                        UpdateProgress();
                                    }
                                    catch (Exception x)
                                    {
                                        HE.WriteLine(x.Message);
                                    }
                                }
                                else
                                {
                                    if (String.Compare(m.Groups[1].Value, "PlateDamagedCode", true) == 0)
                                    {
                                        try
                                        {
                                            if (m.Groups[2].Value.Length != 1) throw new Exception();
                                            PlateDamagedCode = m.Groups[2].Value[0];
                                            specPlateDamagedCode = true;
                                        }
                                        catch (Exception) { }
                                    }
                                    else if (String.Compare(m.Groups[1].Value, "PlateDamaged", true) == 0)
                                    {
                                        try
                                        {
                                            PlateDamaged = Convert.ToInt32(m.Groups[2].Value);
                                        }
                                        catch (Exception) { }
                                    }
                                    else if (String.Compare(m.Groups[1].Value, "GoBackToPlateN", true) == 0)
                                    {
                                        try
                                        {
                                            GoBackToPlateN = Convert.ToInt32(m.Groups[2].Value);

                                            double newZ, currZ;
                                            GetZ(GoBackToPlateN, Plate, out newZ, out currZ);

                                            CancelCalibrations = false;
                                            AbortCurrent = true;
                                            specGoBackToPlateN = true;
                                            if (ProgSettings.Direction == ScanDirection.Upstream)
                                            {
                                                if (/*GoBackToPlateN > Plate*/ newZ < currZ)
                                                {
                                                    GoBackToPlateN = -1;
                                                    CancelCalibrations = false;
                                                    AbortCurrent = false;
                                                    specGoBackToPlateN = false;
                                                }
                                            }
                                            else
                                            {
                                                if (/*GoBackToPlateN < Plate*/ newZ > currZ)
                                                {
                                                    GoBackToPlateN = -1;
                                                    CancelCalibrations = false;
                                                    AbortCurrent = false;
                                                    specGoBackToPlateN = false;
                                                }
                                            }
                                        }
                                        catch (Exception) { }
                                    }
                                    else if (String.Compare(m.Groups[1].Value, "GoBackToPlateNCancelCalibrations", true) == 0)
                                    {
                                        try
                                        {
                                            GoBackToPlateN = Convert.ToInt32(m.Groups[2].Value);

                                            double newZ, currZ;
                                            GetZ(GoBackToPlateN, Plate, out newZ, out currZ);

                                            CancelCalibrations = true;
                                            AbortCurrent = true;
                                            specGoBackToPlateN = true;
                                            if (ProgSettings.Direction == ScanDirection.Upstream)
                                            {
                                                if (/*GoBackToPlateN > Plate*/ newZ < currZ)
                                                {
                                                    GoBackToPlateN = -1;
                                                    CancelCalibrations = false;
                                                    AbortCurrent = false;
                                                    specGoBackToPlateN = false;
                                                }
                                            }
                                            else
                                            {
                                                if (/*GoBackToPlateN < Plate*/ newZ > currZ)
                                                {
                                                    GoBackToPlateN = -1;
                                                    CancelCalibrations = false;
                                                    AbortCurrent = false;
                                                    specGoBackToPlateN = false;
                                                }
                                            }
                                        }
                                        catch (Exception) { }
                                    }
                                    if (specPlateDamagedCode == true && PlateDamagedCode != 'N' && PlateDamaged == Plate && specGoBackToPlateN == false)
                                    {
                                        /*
                                        if (ProgSettings.Direction == ScanDirection.Upstream)
                                        {	
                                            GoBackToPlateN = Plate - 1;
                                        }
                                        else 
                                        {
                                            GoBackToPlateN = Plate + 1;
                                        }
                                        */
                                        GoBackToPlateN = Plate;
                                        specGoBackToPlateN = true;
                                        AbortCurrent = true;
                                        CancelCalibrations = true;
                                    }
                                    HE.WriteLine("PlateDamagedCode = " + PlateDamagedCode);
                                    HE.WriteLine("PlateDamaged = " + PlateDamaged);
                                    HE.WriteLine("GoBackToPlateN = " + GoBackToPlateN);
                                    HE.WriteLine("CancelCalibrations = " + CancelCalibrations);
                                    HE.WriteLine("AbortCurrent = " + AbortCurrent);
                                    if (AbortCurrent)
                                    {
                                        PredictionsDoneEvent.WaitOne();
                                        if (WaitingOnId != 0)
                                        {
                                            SySal.DAQSystem.Drivers.Status waitstatus = SySal.DAQSystem.Drivers.Status.Unknown;
                                            try
                                            {
                                                HE.Abort(WaitingOnId);
                                                waitstatus = HE.Wait(WaitingOnId);
                                            }
                                            catch (Exception)
                                            {
                                                waitstatus = HE.GetStatus(WaitingOnId);
                                            }
                                            if (waitstatus != SySal.DAQSystem.Drivers.Status.Completed && waitstatus != SySal.DAQSystem.Drivers.Status.Failed) throw new Exception("Can't interrupt child process Id " + WaitingOnId);
                                            new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_DELETE_PREDICTIONS(" + StartupInfo.BrickId + ", " + StartupInfo.ProcessOperationId + ", " + Plate + ")", tempConn, null).ExecuteNonQuery();
                                            ScanDone = false;
                                            PredictionsDone = false;
                                        }
                                    }
                                    if (specPlateDamagedCode)
                                        if (SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM VW_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + PlateDamaged, tempConn, null).ExecuteScalar()) == 1)
                                            new SySal.OperaDb.OperaDbCommand("CALL PC_SET_PLATE_DAMAGED(" + StartupInfo.BrickId + ", " + PlateDamaged + ", " + StartupInfo.ProcessOperationId + ", '" + PlateDamagedCode + "')", tempConn, null).ExecuteNonQuery();
                                    if (specGoBackToPlateN)
                                    {
                                        if (ProgSettings.Direction == ScanDirection.Upstream)
                                        {
                                            SySal.OperaDb.OperaDbTransaction Trans = tempConn.BeginTransaction();
                                            try
                                            {
                                                System.Data.DataSet dspl = new System.Data.DataSet();
                                                new SySal.OperaDb.OperaDbDataAdapter("SELECT /*+INDEX_ASC(TB_PLATES PK_PLATES) */ ID, Z FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND Z <= (SELECT  /*+INDEX(TB_PLATES PK_PLATES) */ Z FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + GoBackToPlateN + ") ORDER BY Z ASC", tempConn, null).Fill(dspl);
                                                int newplate = GoBackToPlateN;
                                                foreach (System.Data.DataRow drpl in dspl.Tables[0].Rows)
                                                {
                                                    new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_DELETE_PREDICTIONS(" + StartupInfo.BrickId + ", " + StartupInfo.ProcessOperationId + ", " + drpl[0].ToString() + ")", tempConn, null).ExecuteNonQuery();
                                                    newplate = SySal.OperaDb.Convert.ToInt32(drpl[0]);
                                                    if (CancelCalibrations)
                                                    {
                                                        System.Data.DataSet dsi = new System.Data.DataSet();
                                                        new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_PROCESSOPERATION FROM TB_PLATE_CALIBRATIONS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PLATE = " + newplate, tempConn, null).Fill(dsi);
                                                        foreach (System.Data.DataRow dri in dsi.Tables[0].Rows)
                                                        {
                                                            long icop = SySal.OperaDb.Convert.ToInt64(dri[0]);
                                                            int pos = IgnoreCalibrationList.BinarySearch(icop);
                                                            if (pos < 0) IgnoreCalibrationList.Insert(~pos, icop);
                                                        }
                                                    }
                                                }
                                                Trans.Commit();
                                                Plate = newplate;
                                                ScanDone = false;
                                                IntercalibrationDone = false;
                                                PredictionsDone = false;
                                            }
                                            catch (Exception x)
                                            {
                                                HE.WriteLine("Error during SQL transaction: " + x.Message);
                                                Trans.Rollback();
                                            }
                                        }
                                        else
                                        {
                                            SySal.OperaDb.OperaDbTransaction Trans = tempConn.BeginTransaction();
                                            try
                                            {
                                                System.Data.DataSet dspl = new System.Data.DataSet();
                                                new SySal.OperaDb.OperaDbDataAdapter("SELECT /*+INDEX_ASC(TB_PLATES PK_PLATES) */ ID, Z FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND Z >= (SELECT  /*+INDEX(TB_PLATES PK_PLATES) */ Z FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + GoBackToPlateN + ") ORDER BY Z DESC", tempConn, null).Fill(dspl);
                                                int newplate = GoBackToPlateN;
                                                foreach (System.Data.DataRow drpl in dspl.Tables[0].Rows)
                                                {
                                                    new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_DELETE_PREDICTIONS(" + StartupInfo.BrickId + ", " + StartupInfo.ProcessOperationId + ", " + drpl[0].ToString() + ")", tempConn, null).ExecuteNonQuery();
                                                    newplate = SySal.OperaDb.Convert.ToInt32(drpl[0]);
                                                    if (CancelCalibrations)
                                                    {
                                                        System.Data.DataSet dsi = new System.Data.DataSet();
                                                        new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_PROCESSOPERATION FROM TB_PLATE_CALIBRATIONS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PLATE = " + newplate, tempConn, null).Fill(dsi);
                                                        foreach (System.Data.DataRow dri in dsi.Tables[0].Rows)
                                                        {
                                                            long icop = SySal.OperaDb.Convert.ToInt64(dri[0]);
                                                            int pos = IgnoreCalibrationList.BinarySearch(icop);
                                                            if (pos < 0) IgnoreCalibrationList.Insert(~pos, icop);
                                                        }
                                                    }
                                                }
                                                Trans.Commit();
                                                Plate = newplate;
                                                ScanDone = false;
                                                IntercalibrationDone = false;
                                                PredictionsDone = false;
                                            }
                                            catch (Exception)
                                            {
                                                Trans.Rollback();
                                            }
                                        }
                                    }
                                    if (AbortCurrent)
                                    {
                                        ReloadStatus = true;
                                        WaitingOnId = 0;
                                    }
                                    UpdateProgress();
                                }
                            }
                        }
                    }
                }
                catch (Exception x)
                {
                    HE.WriteLine("Error processing interrupt: " + x.Message);
                }
                finally
                {
                    tempConn.Close();
                }
                HE.LastProcessedInterruptId = nextint.Id;
            }
        }

        #endregion

        #region Recalibration usage

        static void ExecRecomputeCalibration()
        {
            ProgressInfo = HE.ProgressInfo;
            long CurrentOperationId = -1, PreviousOperationId = -1;
            int CurrentPlate = -1, PreviousPlate = -1;
            double CurrentZ = 0.0, PreviousZ = 0.0, OldZ = 0.0;
            try
            {
                //Conn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
                Conn.Open();
                SySal.OperaDb.Schema.DB = Conn;
                SySal.DAQSystem.Scanning.IntercalibrationInfo CurrentIntercal = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
                SySal.DAQSystem.Scanning.IntercalibrationInfo TempIntercal = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
                SySal.DAQSystem.Scanning.IntercalibrationInfo DiffIntercal = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
                SySal.DAQSystem.Scanning.IntercalibrationInfo OldIntercal = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
                SySal.DAQSystem.Scanning.IntercalibrationInfo PreviousIntercal = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
                SySal.OperaDb.Schema.TB_EVENTBRICKS evb = SySal.OperaDb.Schema.TB_EVENTBRICKS.SelectPrimaryKey(StartupInfo.BrickId, SySal.OperaDb.Schema.OrderBy.None);
                evb.Row = 0;
                PreviousIntercal.RX = OldIntercal.RX = CurrentIntercal.RX = (evb._MAXX + evb._MINX) * 0.5 - SySal.OperaDb.Convert.ToDouble(evb._ZEROX);
                PreviousIntercal.RY = OldIntercal.RY = CurrentIntercal.RY = (evb._MAXY + evb._MINY) * 0.5 - SySal.OperaDb.Convert.ToDouble(evb._ZEROY);
                PreviousIntercal.MXX = PreviousIntercal.MYY = 1.0;
                PreviousIntercal.MXY = PreviousIntercal.MYX = 0.0;
                PreviousIntercal.TX = PreviousIntercal.TY = PreviousIntercal.TZ = 1.0;
                TempIntercal = PreviousIntercal;
                switch (ProgSettings.Direction)
                {
                    case ScanDirection.Upstream: CurrentPlate = MaxPlate; CurrentZ = MaxZ; break;
                    case ScanDirection.Downstream: CurrentPlate = MinPlate; CurrentZ = MinZ; break;
                }
                int TotalPlates = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId, Conn, null).ExecuteScalar());
                long LastScanbackId = 0;
                try
                {
                    LastScanbackId = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT TB_PROC_OPERATIONS.ID FROM TB_PROC_OPERATIONS INNER JOIN TB_PROGRAMSETTINGS ON (TB_PROGRAMSETTINGS.ID = TB_PROC_OPERATIONS.ID_PROGRAMSETTINGS AND ID_EVENTBRICK = " + StartupInfo.BrickId + " AND SUCCESS = 'Y' AND LOWER(EXECUTABLE) = 'simplescanback2driver.exe') ORDER BY FINISHTIME DESC", Conn, null).ExecuteScalar());
                }
                catch (Exception x)
                {
                    throw new Exception("Apparently no SimpleScanback3Driver process has ever been completed on this brick. Please make sure you have not used other scanback/scanforth drivers.\r\nMessage:\r\n" + x.Message);
                }
                SySal.OperaDb.OperaDbTransaction trans = Conn.BeginTransaction();
                try
                {
                    while (CurrentPlate >= 0)
                    {
                        System.Data.DataSet dsp = new System.Data.DataSet();
                        new SySal.OperaDb.OperaDbDataAdapter("SELECT DISTINCT ID_PROCESSOPERATION FROM TB_ZONES WHERE (ID_EVENTBRICK, ID_PLATE, ID) in (SELECT ID_EVENTBRICK, ID_PLATE, ID_ZONE FROM TB_SCANBACK_PREDICTIONS WHERE (ID_EVENTBRICK, ID_PLATE, ID_PATH) IN (SELECT " + StartupInfo.BrickId + ", " + CurrentPlate + ", ID FROM TB_SCANBACK_PATHS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PROCESSOPERATION = " + LastScanbackId + "))", Conn, null).Fill(dsp);
                        if (dsp.Tables[0].Rows.Count < 1) throw new Exception("No prediction scan done on plate " + CurrentPlate + ".");
                        if (dsp.Tables[0].Rows.Count > 1) throw new Exception("More than one prediction scan done on plate " + CurrentPlate + ". Ambiguity cannot be solved.");
                        string MarkSet = new SySal.OperaDb.OperaDbCommand("SELECT MARKSET FROM TB_PROGRAMSETTINGS WHERE ID = (SELECT ID_PROGRAMSETTINGS FROM TB_PROC_OPERATIONS WHERE ID = " + dsp.Tables[0].Rows[0].ToString() + ")", Conn, null).ExecuteScalar().ToString();
                        CurrentOperationId = SySal.OperaDb.Convert.ToInt64(dsp.Tables[0].Rows[0][0]);
                        long CurrentCalibrationId = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT NVL(ID_CALIBRATION_OPERATION, 0) FROM TB_PROC_OPERATIONS WHERE ID = " + CurrentOperationId, Conn, null).ExecuteScalar());
                        if (CurrentCalibrationId <= 0)
                        {
                            CurrentZ = SySal.OperaDb.Convert.ToDouble(new SySal.OperaDb.OperaDbCommand("SELECT Z FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + CurrentPlate, Conn, null).ExecuteScalar());
                            CurrentIntercal.MXX = CurrentIntercal.MYY = 1.0;
                            CurrentIntercal.MYX = CurrentIntercal.MXY = 0.0;
                            CurrentIntercal.TX = CurrentIntercal.TY = CurrentIntercal.TZ = 1.0;
                            PreviousIntercal = CurrentIntercal;
                        }
                        else
                        {
                            SySal.OperaDb.Schema.TB_PLATE_CALIBRATIONS plc = SySal.OperaDb.Schema.TB_PLATE_CALIBRATIONS.SelectPrimaryKey(StartupInfo.BrickId, CurrentPlate, CurrentCalibrationId, SySal.OperaDb.Schema.OrderBy.None);
                            plc.Row = 0;
                            OldIntercal.MXX = plc._MAPXX;
                            OldIntercal.MXY = plc._MAPXY;
                            OldIntercal.MYX = plc._MAPYX;
                            OldIntercal.MYY = plc._MAPYY;
                            OldIntercal.TX = plc._MAPDX;
                            OldIntercal.TY = plc._MAPDY;
                            OldIntercal.TZ = 0.0;
                            OldZ = plc._Z;
                        }
                        System.Data.DataSet dstk = new System.Data.DataSet();
                        string wherestr = (ProgSettings.WhereClauseForRecalibrationSelection == null || ProgSettings.WhereClauseForRecalibrationSelection.Length == 0) ? "" : (" AND " + ProgSettings.WhereClauseForRecalibrationSelection.Replace("_BRICK_", StartupInfo.BrickId.ToString()).Replace("_PLATE_", CurrentPlate.ToString()));
                        if (PreviousPlate < 0)
                        {
                            new SySal.OperaDb.OperaDbDataAdapter("select idb, pplate, ppath, pzone, pcand, posx, posy, slopex, slopey, grains, areasum, sigma from " +
                                "(select id_eventbrick as idb, id_plate as pplate, id_path as ppath, id_zone as pzone, id_candidate as pcand from tb_scanback_predictions where (id_eventbrick, id_plate, id_path) in (select " + StartupInfo.BrickId + ", " + CurrentPlate + ", id from tb_scanback_paths where id_eventbrick = " + StartupInfo.BrickId + " and id_processoperation = " + LastScanbackId + " ) and id_candidate is not null " + wherestr + ") " +
                                "inner join tb_mipbasetracks on (id_eventbrick = idb and id_zone = pzone and id = pcand) ",
                                Conn, trans).Fill(dstk);
                        }
                        else
                        {
                            new SySal.OperaDb.OperaDbDataAdapter("select idb, pplate, ppath, pzone, pcand, pposx, pposy, pslopex, pslopey, grains as pgrains, areasum as pareasum, sigma as psigma, posx as sposx, posy as sposy, slopex as sslopex, slopey as sslopey from " +
                                "(select idb, pplate, ppath, pzone, pcand, splate, szone, scand, posx as pposx, posy as pposy, slopex as pslopex, slopey as pslopey from " +
                                "((select id_eventbrick as idb, id_plate as pplate, id_path as ppath, id_zone as pzone, id_candidate as pcand from tb_scanback_predictions where (id_eventbrick, id_plate, id_path) in (select " + StartupInfo.BrickId + ", " + CurrentPlate + ", id from tb_scanback_paths where id_eventbrick = " + StartupInfo.BrickId + " and id_processoperation = " + LastScanbackId + " ) and id_candidate is not null " + wherestr + ") " +
                                "inner join " +
                                "(select id_plate as splate, id_path as spath, id_zone as szone, id_candidate as scand from tb_scanback_predictions where (id_eventbrick, id_plate, id_path) in (select " + StartupInfo.BrickId + ", " + PreviousPlate + ", id from tb_scanback_paths where id_eventbrick = " + StartupInfo.BrickId + " and id_processoperation = " + LastScanbackId + ") and id_candidate is not null) " +
                                "on (ppath = spath)) " +
                                "inner join tb_mipbasetracks on (id_eventbrick = idb and id_zone = pzone and id = pcand)) " +
                                "inner join tb_mipbasetracks on (id_eventbrick = idb and id_zone = szone and id = scand) ",
                                Conn, trans).Fill(dstk);
                        }
                        System.Data.DataRowCollection dstkrc = dstk.Tables[0].Rows;
                        int i;
                        if (dstkrc.Count < 3) throw new Exception("At least 3 scanback paths are needed to calibrate!");
                        double x, y, ox, oy, sx, sy, osx, osy;
                        if (PreviousPlate >= 0)
                        {
                            double[,] coords = new double[2, dstkrc.Count];
                            double[] dx = new double[dstkrc.Count];
                            double[] dy = new double[dstkrc.Count];
                            double[] res = new double[3];
                            double ccorr = 0.0;
                            for (i = 0; i < dstkrc.Count; i++)
                            {
                                System.Data.DataRow dstkr = dstkrc[i];
                                x = coords[0, i] = SySal.OperaDb.Convert.ToDouble(dstkr[5]) - OldIntercal.RX;
                                y = coords[1, i] = SySal.OperaDb.Convert.ToDouble(dstkr[6]) - OldIntercal.RY;
                                ox = SySal.OperaDb.Convert.ToDouble(dstkr[12]) - PreviousIntercal.RX;
                                oy = SySal.OperaDb.Convert.ToDouble(dstkr[13]) - PreviousIntercal.RY;
                                dx[i] = ox + (OldZ - PreviousZ) * SySal.OperaDb.Convert.ToDouble(dstkr[14]) - x;
                                dy[i] = oy + (OldZ - PreviousZ) * SySal.OperaDb.Convert.ToDouble(dstkr[15]) - y;
                            }
                            NumericalTools.Fitting.MultipleLinearRegression(coords, dx, ref res, ref ccorr);
                            DiffIntercal.TX = res[0];
                            DiffIntercal.MXX = 1.0 + res[1];
                            DiffIntercal.MXY = res[2];
                            NumericalTools.Fitting.MultipleLinearRegression(coords, dy, ref res, ref ccorr);
                            DiffIntercal.TY = res[0];
                            DiffIntercal.MYX = res[1];
                            DiffIntercal.MYY = 1.0 + res[2];
                            TempIntercal.MXX = DiffIntercal.MXX * PreviousIntercal.MXX + DiffIntercal.MXY * PreviousIntercal.MYX;
                            TempIntercal.MXY = DiffIntercal.MXX * PreviousIntercal.MXY + DiffIntercal.MXY * PreviousIntercal.MYY;
                            TempIntercal.MYX = DiffIntercal.MYX * PreviousIntercal.MXX + DiffIntercal.MYY * PreviousIntercal.MYX;
                            TempIntercal.MYY = DiffIntercal.MYX * PreviousIntercal.MXY + DiffIntercal.MYY * PreviousIntercal.MYY;
                            TempIntercal.TX = DiffIntercal.MXX * PreviousIntercal.TX + DiffIntercal.MXY * PreviousIntercal.TY + DiffIntercal.TX;
                            TempIntercal.TY = DiffIntercal.MYX * PreviousIntercal.TX + DiffIntercal.MYY * PreviousIntercal.TY + DiffIntercal.TY;
                            TempIntercal.RX = PreviousIntercal.RX;
                            TempIntercal.RY = PreviousIntercal.RY;
                            CurrentIntercal.MXX = TempIntercal.MXX * OldIntercal.MXX + TempIntercal.MXY * OldIntercal.MYX;
                            CurrentIntercal.MXY = TempIntercal.MXX * OldIntercal.MXY + TempIntercal.MXY * OldIntercal.MYY;
                            CurrentIntercal.MYX = TempIntercal.MYX * OldIntercal.MXX + TempIntercal.MYY * OldIntercal.MYX;
                            CurrentIntercal.MYY = TempIntercal.MYX * OldIntercal.MXY + TempIntercal.MYY * OldIntercal.MYY;
                            CurrentIntercal.TX = TempIntercal.TX + TempIntercal.MXX * OldIntercal.TX + TempIntercal.MXY * OldIntercal.TY;
                            CurrentIntercal.TY = TempIntercal.TY + TempIntercal.MYX * OldIntercal.TX + TempIntercal.MYY * OldIntercal.TY;
                            CurrentZ = OldZ;
                        }
                        else CurrentIntercal = OldIntercal;
                        SySal.OperaDb.Schema.PC_CALIBRATE_PLATE.Call(StartupInfo.BrickId, CurrentPlate, StartupInfo.ProcessOperationId, MarkSet,
                            CurrentZ, CurrentIntercal.MXX, CurrentIntercal.MXY, CurrentIntercal.MYX, CurrentIntercal.MYY, CurrentIntercal.TX, CurrentIntercal.TY);
                        System.DateTime time = System.DateTime.Now;
                        long newzoneid = SySal.OperaDb.Schema.TB_ZONES.Insert(StartupInfo.BrickId, CurrentPlate, StartupInfo.ProcessOperationId, 0, evb._MINX, evb._MAXX, evb._MINY, evb._MAXY, "Virtual Zone", time, time.AddSeconds(1.0), 104, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
                        SySal.OperaDb.Schema.TB_VIEWS.Insert(StartupInfo.BrickId, newzoneid, 1, 1, CurrentZ, CurrentZ, CurrentIntercal.RX, CurrentIntercal.RY);
                        SySal.OperaDb.Schema.TB_VIEWS.Insert(StartupInfo.BrickId, newzoneid, 2, 1, CurrentZ, CurrentZ, CurrentIntercal.RX, CurrentIntercal.RY);
                        SySal.OperaDb.Schema.TB_VIEWS.Flush();
                        i = 0;
                        foreach (System.Data.DataRow dstkr in dstkrc)
                        {
                            ++i;
                            ox = SySal.OperaDb.Convert.ToDouble(dstkr[5]) - TempIntercal.RX;
                            oy = SySal.OperaDb.Convert.ToDouble(dstkr[6]) - TempIntercal.RY;
                            x = TempIntercal.MXX * ox + TempIntercal.MXY * oy + TempIntercal.TX + TempIntercal.RX;
                            y = TempIntercal.MYX * ox + TempIntercal.MYY * oy + TempIntercal.TY + TempIntercal.RY;
                            osx = SySal.OperaDb.Convert.ToDouble(dstkr[7]);
                            osy = SySal.OperaDb.Convert.ToDouble(dstkr[8]);
                            sx = TempIntercal.MXX * osx + TempIntercal.MXY * osy;
                            sy = TempIntercal.MYX * osx + TempIntercal.MYY * osy;
                            int tgrains = SySal.OperaDb.Convert.ToInt32(dstkr[9]) / 2;
                            int bgrains = SySal.OperaDb.Convert.ToInt32(dstkr[9]) - tgrains;
                            int tareasum = SySal.OperaDb.Convert.ToInt32(dstkr[10]) * tgrains / (tgrains + bgrains);
                            int bareasum = SySal.OperaDb.Convert.ToInt32(dstkr[10]) - tareasum;
                            double sigma = SySal.OperaDb.Convert.ToDouble(dstkr[11]);
                            SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Insert(StartupInfo.BrickId, newzoneid, 1, i, x, y, sx, sy, tgrains, tareasum, System.DBNull.Value, sigma, 1);
                            SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Insert(StartupInfo.BrickId, newzoneid, 2, i, x, y, sx, sy, bgrains, bareasum, System.DBNull.Value, sigma, 1);
                            SySal.OperaDb.Schema.TB_MIPBASETRACKS.Insert(StartupInfo.BrickId, newzoneid, i, x, y, sx, sy, tgrains + bgrains, tareasum + bareasum, System.DBNull.Value, sigma, 1, i, 2, i);
                            SySal.OperaDb.Schema.TB_PATTERN_MATCH.Insert(StartupInfo.BrickId, SySal.OperaDb.Convert.ToInt64(dstkr[3]), newzoneid, SySal.OperaDb.Convert.ToInt32(dstkr[4]), i, StartupInfo.ProcessOperationId);
                        }
                        SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Flush();
                        SySal.OperaDb.Schema.TB_MIPBASETRACKS.Flush();
                        SySal.OperaDb.Schema.TB_PATTERN_MATCH.Flush();
                        HE.WriteLine("New calibration for plate " + CurrentPlate + ":");
                        HE.WriteLine("MXX = " + CurrentIntercal.MXX.ToString("F7", System.Globalization.CultureInfo.InvariantCulture) + "   was: " + OldIntercal.MXX.ToString("F7", System.Globalization.CultureInfo.InvariantCulture) + "  diff: " + DiffIntercal.MXX.ToString("F7", System.Globalization.CultureInfo.InvariantCulture) + "  temp: " + TempIntercal.MXX.ToString("F7", System.Globalization.CultureInfo.InvariantCulture));
                        HE.WriteLine("MXY = " + CurrentIntercal.MXY.ToString("F7", System.Globalization.CultureInfo.InvariantCulture) + "   was: " + OldIntercal.MXY.ToString("F7", System.Globalization.CultureInfo.InvariantCulture) + "  diff: " + DiffIntercal.MXY.ToString("F7", System.Globalization.CultureInfo.InvariantCulture) + "  temp: " + TempIntercal.MXY.ToString("F7", System.Globalization.CultureInfo.InvariantCulture));
                        HE.WriteLine("MYX = " + CurrentIntercal.MYX.ToString("F7", System.Globalization.CultureInfo.InvariantCulture) + "   was: " + OldIntercal.MYX.ToString("F7", System.Globalization.CultureInfo.InvariantCulture) + "  diff: " + DiffIntercal.MYX.ToString("F7", System.Globalization.CultureInfo.InvariantCulture) + "  temp: " + TempIntercal.MYX.ToString("F7", System.Globalization.CultureInfo.InvariantCulture));
                        HE.WriteLine("MYY = " + CurrentIntercal.MYY.ToString("F7", System.Globalization.CultureInfo.InvariantCulture) + "   was: " + OldIntercal.MYY.ToString("F7", System.Globalization.CultureInfo.InvariantCulture) + "  diff: " + DiffIntercal.MYY.ToString("F7", System.Globalization.CultureInfo.InvariantCulture) + "  temp: " + TempIntercal.MYY.ToString("F7", System.Globalization.CultureInfo.InvariantCulture));
                        HE.WriteLine("TX = " + CurrentIntercal.TX.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "   was: " + OldIntercal.TX.ToString("F7", System.Globalization.CultureInfo.InvariantCulture) + "  diff: " + DiffIntercal.TX.ToString("F7", System.Globalization.CultureInfo.InvariantCulture) + "  temp: " + TempIntercal.TX.ToString("F7", System.Globalization.CultureInfo.InvariantCulture));
                        HE.WriteLine("TY = " + CurrentIntercal.TY.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "   was: " + OldIntercal.TY.ToString("F7", System.Globalization.CultureInfo.InvariantCulture) + "  diff: " + DiffIntercal.TY.ToString("F7", System.Globalization.CultureInfo.InvariantCulture) + "  temp: " + TempIntercal.TY.ToString("F7", System.Globalization.CultureInfo.InvariantCulture));
                        PreviousZ = CurrentZ;
                        PreviousPlate = CurrentPlate;
                        PreviousIntercal = TempIntercal;
                        PreviousOperationId = CurrentOperationId;
                        int RemainingPlates = 0;
                        TotalPlates = 0;
                        if (ProgSettings.Direction == ScanDirection.Upstream)
                        {
                            RemainingPlates = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND Z < (SELECT Z FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + CurrentPlate + ")", Conn, trans).ExecuteScalar());
                            TotalPlates = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId, Conn, trans).ExecuteScalar());
                            object o = new SySal.OperaDb.OperaDbCommand("select id from (select /*+index_asc(tb_plates pk_plates) */ ID, row_number() over (order by z desc) as rnum from vw_plates where id_eventbrick = "
                                + StartupInfo.BrickId + " and z < (select /*+index(tb_plates pk_plates) */ z from tb_plates where id_eventbrick = " + StartupInfo.BrickId + " and id = " + CurrentPlate + ") and damaged = 'N') where rnum = 1",
                                Conn, null).ExecuteScalar();
                            if (o == System.DBNull.Value || o == null) CurrentPlate = -1;
                            else CurrentPlate = SySal.OperaDb.Convert.ToInt32(o);
                        }
                        else
                        {
                            RemainingPlates = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND Z > (SELECT Z FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + CurrentPlate + ")", Conn, trans).ExecuteScalar());
                            TotalPlates = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId, Conn, trans).ExecuteScalar());
                            object o = new SySal.OperaDb.OperaDbCommand("select id from (select /*+index_asc(tb_plates pk_plates) */ ID, row_number() over (order by z asc) as rnum from vw_plates where id_eventbrick = "
                                + StartupInfo.BrickId + " and z > (select /*+index(tb_plates pk_plates) */ z from tb_plates where id_eventbrick = " + StartupInfo.BrickId + " and id = " + CurrentPlate + ") and damaged = 'N') where rnum = 1",
                                Conn, null).ExecuteScalar();
                            if (o == System.DBNull.Value || o == null) CurrentPlate = -1;
                            else CurrentPlate = SySal.OperaDb.Convert.ToInt32(o);
                        }
                        System.DateTime Current = System.DateTime.Now;
                        ProgressInfo.FinishTime = Current.AddSeconds((Current - ProgressInfo.StartTime).TotalSeconds * RemainingPlates / TotalPlates);
                        ProgressInfo.Progress = 1.0 - (double)RemainingPlates / (double)TotalPlates;
                        HE.ProgressInfo = ProgressInfo;
                    }
                    trans.Commit();
                    HE.WriteLine("Transaction committed, data written.");
                }
                catch (Exception x)
                {
                    if (trans != null) trans.Rollback();
                    throw x;
                }
            }
            finally
            {
                Conn.Close();
            }
        }
        #endregion

        static void CheckVariables(NumericalTools.CStyleParsedFunction f)
        {
            if (f.ParameterList.Length == 0) throw new Exception("A propagation function cannot be a constant.");
            foreach (string s in f.ParameterList)
            {
                if (String.Compare(s, "LP", true) == 0) continue;
                if (String.Compare(s, "LS", true) == 0) continue;
                if (String.Compare(s, "LZ", true) == 0) continue;
                if (String.Compare(s, "FP", true) == 0) continue;
                if (String.Compare(s, "FS", true) == 0) continue;
                if (String.Compare(s, "Z", true) == 0) continue;
                if (String.Compare(s, "F", true) == 0) continue;
                if (String.Compare(s, "N", true) == 0) continue;
                if (String.Compare(s, "A", true) == 0) continue;
                if (String.Compare(s, "S", true) == 0) continue;
                throw new Exception("Unknown variable \"" + s + "\".");
            }
        }

        #region IWebApplication Members

        public string ApplicationName
        {
            get { return "SimpleScanback3Driver"; }
        }

        public SySal.Web.ChunkedResponse HttpGet(SySal.Web.Session sess, string page, params string[] queryget)
        {
            return HttpPost(sess, page, queryget);
        }

        const string ContinueBtn = "continue";
        const string CloseBtn = "close";
        const string GoBackBtn = "gbk";
        const string GoBackCancelCalibsBtn = "gbkcc";
        const string GoBackPlateText = "gbkt";
        const string PlateDamagedBtn = "pdmg";
        const string PlateDamagedText = "pdmgt";
        const string PlateDamagedCodeText = "pdmgct";
        const string LoadPredictionsBtn = "lpred";
        const string LoadPredictionsText = "lpredt";
        const string IgnoreCalibrationsBtn = "ical";
        const string IgnoreCalibrationsText = "icalt";
        const string ForceCalibrationsBtn = "fcal";
        const string ForceCalibrationsText = "fcalt";
        const string SetPathBtn = "spath";
        const string SetPathText = "spatht";

        /// <summary>
        /// Processes HTTP POST method calls.
        /// </summary>
        /// <param name="sess">the user session.</param>
        /// <param name="page">the page requested (ignored).</param>
        /// <param name="postfields">commands sent to the page.</param>
        /// <returns>an HTML string with the page to be shown.</returns>
        public SySal.Web.ChunkedResponse HttpPost(SySal.Web.Session sess, string page, params string[] postfields)
        {
            string xctext = null;
            if (postfields != null)
            {
                bool gobackset = false;
                bool gobackcancelset = false;
                int gobackplate = -1;
                bool platedamagedset = false;
                int platedamagedplate = -1;
                string platedamagedcode = "";
                bool loadpredset = false;
                string[] preds = new string[0];
                bool ignorecalset = false;
                string[] ignorecals = new string[0];
                bool forcecalset = false;
                string[] forcecals = new string[0];
                bool setpathset = false;
                string[] setpaths = new string[0];
                Interrupt i = new Interrupt();
                i.Id = 0;
                foreach (string s in postfields)
                {
                    if (s.StartsWith(ContinueBtn))
                    {
                        i.Data = "Continue";
                    }
                    else if (s.StartsWith(CloseBtn))
                    {
                        i.Data = "StandBy";
                    }
                    else if (s.StartsWith(GoBackBtn + "="))
                    {
                        gobackset = true;
                    }
                    else if (s.StartsWith(GoBackCancelCalibsBtn + "="))
                    {
                        gobackcancelset = true;
                    }
                    else if (s.StartsWith(GoBackPlateText + "="))
                    {
                        try
                        {
                            gobackplate = Convert.ToInt32(s.Substring(GoBackPlateText.Length + 1));
                        }
                        catch (Exception) { }
                    }
                    else if (s.StartsWith(PlateDamagedBtn + "="))
                    {
                        platedamagedset = true;
                    }
                    else if (s.StartsWith(PlateDamagedText + "="))
                    {
                        try
                        {
                            platedamagedplate = Convert.ToInt32(s.Substring(PlateDamagedText.Length + 1));
                        }
                        catch (Exception) { }
                    }
                    else if (s.StartsWith(PlateDamagedCodeText + "="))
                    {
                        try
                        {
                            platedamagedcode = s.Substring(PlateDamagedCodeText.Length + 1);
                        }
                        catch (Exception) { }
                    }
                    else if (s.StartsWith(LoadPredictionsBtn + "="))
                    {
                        loadpredset = true;
                    }
                    else if (s.StartsWith(LoadPredictionsText + "="))
                    {
                        preds = SySal.Web.WebServer.URLDecode(s.Substring(LoadPredictionsText.Length + 1)).Split('\n');
                    }
                    else if (s.StartsWith(IgnoreCalibrationsBtn + "="))
                    {
                        ignorecalset = true;
                    }
                    else if (s.StartsWith(IgnoreCalibrationsText + "="))
                    {
                        ignorecals = SySal.Web.WebServer.URLDecode(s.Substring(IgnoreCalibrationsText.Length + 1)).Split('\n');
                    }
                    else if (s.StartsWith(ForceCalibrationsBtn + "="))
                    {
                        forcecalset = true;
                    }
                    else if (s.StartsWith(ForceCalibrationsText + "="))
                    {
                        forcecals = SySal.Web.WebServer.URLDecode(s.Substring(ForceCalibrationsText.Length + 1)).Split('\n');
                    }
                    else if (s.StartsWith(SetPathBtn + "="))
                    {
                        setpathset = true;
                    }
                    else if (s.StartsWith(SetPathText + "="))
                    {
                        setpaths = SySal.Web.WebServer.URLDecode(s.Substring(SetPathText.Length + 1)).Split('\n');
                    }
                }
                if (gobackcancelset && gobackplate > 0)
                {
                    i.Data = "GoBackToPlateNCancelCalibrations " + gobackplate;
                }
                else if (gobackset && gobackplate > 0)
                {
                    i.Data = "GoBackToPlateN " + gobackplate;
                }
                else if (platedamagedset && platedamagedplate > 0 && platedamagedcode.Length > 0)
                {
                    i.Data = "PlateDamaged " + platedamagedplate + ", PlateDamagedCode " + platedamagedcode;
                }
                else if (loadpredset && preds.Length > 0)
                {
                    i.Data = "Paths " + preds.Length;
                    foreach (string p in preds)
                        i.Data += "; " + p;
                }
                else if (ignorecalset && ignorecals.Length > 0)
                {
                    i.Data = "IgnoreCalibrations " + ignorecals.Length;
                    foreach (string p in ignorecals)
                        i.Data += "; " + p;
                }
                else if (forcecalset && forcecals.Length > 0)
                {
                    i.Data = "ForceCalibrations " + forcecals.Length;
                    foreach (string p in forcecals)
                        i.Data += "; " + p;
                }
                else if (setpathset && setpaths.Length > 0)
                {
                    i.Data = "SetPaths " + setpaths.Length;
                    foreach (string p in setpaths)
                        i.Data += "; " + p;
                }
                if (i.Data != null)
                    try
                    {
                        NotifyInterrupt(i);
                    }
                    catch (Exception x)
                    {
                        xctext = x.ToString();
                    }
            }
            string html = "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\" \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\r\n" +
                "<html xmlns=\"http://www.w3.org/1999/xhtml\" >\r\n" +
                "<head>\r\n" +
                "    <meta http-equiv=\"pragma\" content=\"no-cache\">\r\n" +
                "    <meta http-equiv=\"EXPIRES\" content=\"0\" />\r\n" +
                "    <title>SimpleScanback3Driver - " + StartupInfo.BrickId + "/" + StartupInfo.ProcessOperationId + "</title>\r\n" +
                "    <style type=\"text/css\">\r\n" +
                "    th { font-family: Arial,Helvetica; font-size: 12; color: white; background-color: teal; text-align: center; font-weight: bold }\r\n" +
                "    td { font-family: Arial,Helvetica; font-size: 12; color: navy; background-color: white; text-align: right; font-weight: normal }\r\n" +
                "    p {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: left; font-weight: normal }\r\n" +
                "    div {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                "    </style>\r\n" +
                "</head>\r\n" +
                "<body>\r\n" +
                " <div>SimpleScanback3Driver = " + StartupInfo.ProcessOperationId + "<br>Brick = " + StartupInfo.BrickId + "<br>Plate = " + Plate + "<br>CanRun = " + CanRun.WaitOne(0) + "<br>Direction = " + ProgSettings.Direction + "</div>\r\n<hr>\r\n" +
                ((xctext != null) ? "<div>Interrupt Error:<br><font color=\"red\">" + SySal.Web.WebServer.HtmlFormat(xctext) + "<font></div>\r\n" : "") +
                " <form action=\"" + page + "\" method=\"post\" enctype=\"application/x-www-form-urlencoded\">\r\n" +
                "  <div>\r\n" +
                "   <input id=\"" + ContinueBtn + "\" name=\"" + ContinueBtn + "\" type=\"submit\" value=\"Continue\"/>&nbsp;<input id=\"" + CloseBtn + "\" name=\"" + CloseBtn + "\" type=\"submit\" value=\"Stand by\"/><br>\r\n" +
                "   <input id=\"" + GoBackBtn + "\" name=\"" + GoBackBtn + "\" type=\"submit\" value=\"Go Back to Plate\"/>&nbsp;<input id=\"" + GoBackCancelCalibsBtn + "\" name=\"" + GoBackCancelCalibsBtn + "\" type=\"submit\" value=\"Go Back to Plate and Cancel Calibrations\"/>&nbsp;<input id=\"" + GoBackPlateText + "\" maxlength=\"3\" name=\"" + GoBackPlateText + "\" size=\"3\" type=\"text\" />\r\n" +
                "   <input id=\"" + PlateDamagedBtn + "\" name=\"" + PlateDamagedBtn + "\" type=\"submit\" value=\"Mark Plate Damaged\"/>&nbsp;<input id=\"" + PlateDamagedText + "\" maxlength=\"3\" name=\"" + PlateDamagedText + "\" size=\"3\" type=\"text\" value=\"" + Plate + "\" />&nbsp;<input id=\"" + PlateDamagedCodeText + "\" maxlength=\"3\" name=\"" + PlateDamagedCodeText + "\" size=\"3\" type=\"text\" value=\"N\" />\r\n" +
                "   <input id=\"" + LoadPredictionsBtn + "\" name=\"" + LoadPredictionsBtn + "\" type=\"submit\" value=\"Load Predictions\"/>&nbsp;<textarea id=\"" + LoadPredictionsText + "\" name=\"" + LoadPredictionsText + "\" size=\"3\" rows=\"4\" cols=\"20\" /></textarea>\r\n" +
                "   <input id=\"" + IgnoreCalibrationsBtn + "\" name=\"" + IgnoreCalibrationsBtn + "\" type=\"submit\" value=\"Ignore Calibrations\"/>&nbsp;<textarea id=\"" + IgnoreCalibrationsText + "\" name=\"" + IgnoreCalibrationsText + "\" size=\"3\" rows=\"4\" cols=\"20\" /></textarea>\r\n" +
                "   <input id=\"" + ForceCalibrationsBtn + "\" name=\"" + ForceCalibrationsBtn + "\" type=\"submit\" value=\"Ignore Calibrations\"/>&nbsp;<textarea id=\"" + ForceCalibrationsText + "\" name=\"" + ForceCalibrationsText + "\" size=\"3\" rows=\"4\" cols=\"20\" /></textarea>\r\n" +
                "   <input id=\"" + SetPathBtn + "\" name=\"" + SetPathBtn + "\" type=\"submit\" value=\"Set Paths\"/>&nbsp;<textarea id=\"" + SetPathText + "\" name=\"" + SetPathText + "\" size=\"3\" rows=\"4\" cols=\"20\" /></textarea>\r\n" +
                "  </div>\r\n" +
                " </form>\r\n" +
                "</body>\r\n" +
                "</html>";

            return new SySal.Web.HTMLResponse(html);
        }

        public bool ShowExceptions
        {
            get { return true; }
        }

        #endregion
    }
}
