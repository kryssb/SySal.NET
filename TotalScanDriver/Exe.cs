using System;
using SySal;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;
using SySal.DAQSystem.Drivers;
using System.Xml;
using System.Xml.Serialization;

namespace SySal.DAQSystem.Drivers.TotalScanDriver
{
    class ExtBoxDesc : SySal.DAQSystem.Drivers.BoxDesc
    {
        public int RefPlate;
        public long ReuseVolumeId;
        public int[] PlatesChosen;
    }

	/// <summary>
	/// Scanning direction.
	/// </summary>
	[Serializable]
	public enum ScanDirection
	{
		/// <summary>
		/// Scan towards the upstream direction.
		/// </summary>
		Upstream,
		/// <summary>
		/// Scan towards the downstream direction.
		/// </summary>
		Downstream
	}

	/// <summary>
	/// Source for input.
	/// </summary>
	[Serializable]
	public enum InputSource
	{
		/// <summary>
		/// Volumes are built around stopping points of scanback/scanforth paths using the last slope.
		/// Reads TB_B_SBPATHS_VOLUMES.
		/// </summary>
		ScanbackPath,
		/// <summary>
		/// Volumes are built around stopping points of scanback/scanforth paths using a fixed primary beam slope.
		/// Reads TB_B_SBPATHS_VOLUMES.
		/// </summary>
		ScanbackPathFixedPrimarySlope,
		/// <summary>
		/// Set up volumes to extend already existing volume track.
		/// <b>Warning: this is not yet supported!</b>
		/// </summary>
		VolumeTrack,
		/// <summary>
		/// Set up volumes from an ASCII n-tuple given through an interrupt.
		/// </summary>
		Interrupt
	}

    /// <summary>
    /// Scheme to apply to reduce the number of plates to scan.
    /// </summary>
    [Serializable]
    public enum SliceReductionScheme 
    { 
        /// <summary>
        /// Scan all plates.
        /// </summary>
        NoReduction, 
        /// <summary>
        /// Scan one plate every two (odd plates).
        /// </summary>
        EveryTwo, 
        /// <summary>
        /// Scan one plate every four (1,5,9,...).
        /// </summary>
        EveryFour, 
        /// <summary>
        /// Scan one plate every two (odd plates), but always include the first two and last two plates of the brick.
        /// </summary>
        EveryTwoOrExit, 
        /// <summary>
        /// Scan one plate every four, but always include the first four and last four plates of the brick.
        /// </summary>
        EveryFourOrExit
    }

	/// <summary>
	/// Volume creation parameters.
	/// </summary>
	[Serializable]
	public class VolumeCreation
	{
		/// <summary>
		/// Source for input.
		/// </summary>
		public InputSource Source;
		/// <summary>
		/// Sets the number of plates to be scanned downstream of the interesting plate.
		/// </summary>
		public uint DownstreamPlates;
		/// <summary>
		/// Sets the number of plates to be scanned upstream of the interesting plate.
		/// </summary>
		public uint UpstreamPlates;
		/// <summary>
		/// A formula to specify the zone width (size in X direction). The formula can be a constant, or it can contain the following parameters:
		/// <list type="table">
		/// <item><term><c>DPLATE</c></term><description>The difference between the interesting plate Id and a generic plate Id. E.g.: if the interesting plate is 48, and the current plate is 46, <c>DPLATE</c> = -2.</description></item>
		/// <item><term><c>DZ</c></term><description>The difference between the interesting plate Z and a generic plate Z. E.g.: if the interesting plate is 48 with = 20000, and the current plate is 46 with Z = 17400, <c>DZ</c> = -2600.</description></item>
		/// </list>
		/// If <c>HeightFormula</c> is not specified, <c>WidthFormula</c> is used as a default.
		/// </summary>
		public string WidthFormula;
		/// <summary>
		/// A formula to specify the zone height (size in Y direction). The formula can be a constant, or it can contain the following parameters:
		/// <list type="table">
		/// <item><term><c>DPLATE</c></term><description>The difference between the interesting plate Id and a generic plate Id. E.g.: if the interesting plate is 48, and the current plate is 46, <c>DPLATE</c> = -2.</description></item>
		/// <item><term><c>DZ</c></term><description>The difference between the interesting plate Z and a generic plate Z. E.g.: if the interesting plate is 48 with = 20000, and the current plate is 46 with Z = 17400, <c>DZ</c> = -2600.</description></item>
		/// </list>
		/// If <c>HeightFormula</c> is not specified, <c>WidthFormula</c> is used as a default.
		/// </summary>
		public string HeightFormula;
		/// <summary>
		/// Slope of the primary beam (only applies to the case where <c>Source = ScanbackPathFixedPrimarySlope</c>).
		/// </summary>
		public SySal.BasicTypes.Vector2 PrimarySlope;
        /// <summary>
        /// if <c>true</c>, it checks that volumes do not cross plate bounds (e.g. plate size).
        /// </summary>
        public bool CheckBounds;
        /// <summary>
        /// When computing bounds, this margin is added to shrink the usable plate size.
        /// </summary>
        public double BoundMargin;
        /// <summary>
        /// The reduction scheme to be used for volume creation based on interrupts.
        /// </summary>
        public SliceReductionScheme ReductionScheme;        
	}

	/// <summary>
	/// Settings for TotalScanDriver.
	/// </summary>
	[Serializable]
	public class TotalScanSettings
	{
		/// <summary>
		/// Program settings Id for Intercalibration.
		/// If this Id is equal to AreaScanConfigId, the driver assumes that the same process operation performs intercalibration as well as scanning.
		/// If intercalibration is simply to be skipped, this Id should be set equal to AreaScanConfigId, <b>not zero</b>.
		/// </summary>
		public long IntercalibrationConfigId;
		/// <summary>
		/// Program settings Id for Area Scanning.
		/// </summary>
		public long AreaScanConfigId;		
		/// <summary>
		/// Scanning direction.
		/// </summary>
		public ScanDirection Direction;
		/// <summary>
		/// Creation mode for volumes.
		/// </summary>
		public VolumeCreation VolumeCreationMode;
		/// <summary>
		/// If <c>true</c>, calibrated plates are re-calibrated if they had been calibrated within a previous volume operation. Valid Calibrations are re-used if this is set to <c>false</c>.
		/// </summary>
		public bool ForceRefreshCalibration;
	}

	/// <summary>
	/// TotalScanDriver executor.
	/// </summary>
	/// <remarks>
	/// <para>TotalScanDriver performs TotalScan throughout a brick.</para>
	/// <para>TotalScan initialization can come from different data sources:
	/// <list type = "table">
	/// <listheader><term>Source</term><description>Explanation</description></listheader>
	/// <item><term>ScanbackPath</term><description>starts scanning around scanback/scanforth interesting points, using the last seen slope.</description></item>
	/// <item><term>ScanbackPathFixedPrimarySlope</term><description>starts scanning around scanback/scanforth interesting points, using the primary beam slope.</description></item>
	/// <item><term>VolumeTrack</term><description>follows the slope of an already existing volume track.</description></item>
	/// <item><term>Interrupt</term><description>reads volumes to be scanned from an interrupt.</description></item>
	/// </list>
	/// The TB_VOLUMES and TB_VOLUME_SLICES tables are used to record the TotalScan process.
	/// </para>
	/// <para>Type: <c>TotalScanDriver /Interrupt &lt;batchmanager&gt; &lt;process operation id&gt; &lt;interrupt string&gt;</c> to send an interrupt message to a running TotalScanDriver process operation.</para>
	/// <para>
	/// Supported Interrupts:
	/// <list type="table">
	/// <item><term><c>PlateDamagedCode &lt;code&gt;</c></term><description>Instructs TotalScanDriver to use the specified code to mark the plate as damaged. The plate must be specified by PlateDamaged. If it's missing, the current plate number is assumed.</description></item>
	/// <item><term><c>PlateDamaged &lt;plate&gt;</c></term><description>Instructs TotalScanDriver to mark the specified plate as damaged. If it's missing, the current plate number is assumed.</description></item>
	/// <item><term><c>GoBackToPlateN &lt;plate&gt;</c></term><description>Instructs TotalScanDriver to go back to the specified plate, keeping intercalibration info.</description></item>
	/// <item><term><c>GoBackToPlateNCancelCalibrations &lt;plate&gt;</c></term><description>Instructs TotalScanDriver to go back to the specified plate, cancelling intercalibration info obtained by daughter operations of the current one. Intercalibrations obtained by previous operations will not be cancelled.</description></item>
	/// <item><term><c>Volumes &lt;number&gt;</c></term><description>Used to provide volumes on startup when the volume source is set to Interrupt in the ProgramSettings. The number sets the number of expected volumes. Volumes are 7-tuples such as <c>VOLUME MINX MAXX MINY MAXY MINPLATE MAXPLATE</c> separated by ';'. The first volume must be preceeded by ';'. No ';' is to be put at the end of the volume string. The volume string may contain any spacers including newlines.</description></item>
	/// </list>
	/// </para>
	/// <para>An example of interrupt for volume specification follows:</para>
	/// <para>
	/// <example>
	/// <code>
	/// Volumes 3;
    /// 1 10204.3 12204.3 14893.2 16893.2 4 12;
    /// 2 11244.3 13244.3 18823.2 20823.2 8 16;
	/// 3 8848.1 1248.5 10848.1 3248.5 23 37
	/// </code>
    /// </example>
    /// </para>
    /// <para>An extended syntax allows setting skewed volumes, adding the id of the plate on which the extents are set and the skewing slopes:</para>
    /// <para>
    /// <example>
    /// <code>
    /// Volumes 2;
    /// 1 10204.3 12204.3 14893.2 16893.2 4 12 5 0.1 0.3;
    /// 2 11244.3 13244.3 18823.2 20823.2 8 16 16 -0.1 0.5;    
    /// </code>
    /// </example>	
    /// </para>
    /// <para>A further extended syntax allows setting skewed volumes, adding the id of the plate on which the extents are set and the skewing slopes, and specifying that data from an existing volume should be reused:</para>
    /// <para>
    /// <example>
    /// <code>
    /// Volumes 2;
    /// 1 10204.3 12204.3 14893.2 16893.2 4 12 5 0.1 0.3 100044959573;
    /// 2 11244.3 13244.3 18823.2 20823.2 8 16 16 -0.1 0.5 100044959577;    
    /// </code>
    /// </example>
    /// </para>
    /// <para>A further extended syntax allows setting skewed volumes, adding the id of the plate on which the extents are set and the skewing slopes, and specifying that single plates from an existing volume should be reused:</para>
    /// <para>
    /// <example>
    /// <code>
    /// Volumes 1;
    /// 1 10204.3 12204.3 14893.2 16893.2 48 57 5 0.1 0.3 100044959573(48 50 51 53 54 56 57);    
    /// </code>
    /// </example>
    /// In this example, one wants to "replace" data for plates 49 and 52, and keep all others. The existing volume is not replaced, but a new one is created.
    /// </para>
    /// <para>Type: <c>TotalScanDriver /EasyInterrupt</c> for a graphical user interface to send interrupts.</para>
	/// <para>
	/// A sample XML configuration for TotalScanDriver follows:
	/// <example>
	/// <code>
	/// &lt;TotalScanSettings&gt;
	///  &lt;IntercalibrationConfigId&gt;80088238&lt;/IntercalibrationConfigId&gt;
	///  &lt;AreaScanConfigId&gt;80088275&lt;/AreaScanConfigId&gt;
	///  &lt;Direction&gt;Upstream&lt;/Direction&gt;
	///  &lt;VolumeCreationMode&gt;
	///   &lt;Source&gt;ScanbackPathFixedPrimarySlope&lt;/Source&gt;
	///   &lt;DownstreamPlates&gt;8&lt;/DownstreamPlates&gt;
	///   &lt;UpstreamPlates&gt;1&lt;/UpstreamPlates&gt;
	///   &lt;WidthFormula&gt;5000&lt;/WidthFormula&gt;
	///   &lt;PrimarySlope&gt;
	///    &lt;X&gt;0.05&lt;/X&gt;
	///    &lt;Y&gt;-0.01&lt;/Y&gt;
	///   &lt;/PrimarySlope&gt;
	///  &lt;/VolumeCreationMode&gt;
	/// &lt;/TotalScanSettings&gt;	
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
			strw.WriteLine("TotalScanDriver");
			strw.WriteLine("--------------");
			strw.WriteLine("TotalScanDriver performs TotalScan throughout a brick.");
			strw.WriteLine("--------------");
			strw.WriteLine("TotalScan initialization can come from different data sources:");
			strw.WriteLine("ScanbackPath -> starts scanning around scanback/scanforth interesting points, using the last seen slope.");
			strw.WriteLine("ScanbackPathFixedPrimarySlope -> starts scanning around scanback/scanforth interesting points, using the primary beam slope.");
			strw.WriteLine("VolumeTrack -> follows the slope of an already existing volume track.");
			strw.WriteLine("Interrupt -> reads volumes to be scanned from an interrupt.");
			strw.WriteLine();
			strw.WriteLine("The TB_VOLUMES and TB_VOLUME_SLICES tables are used to record the TotalScan process.");
			strw.WriteLine();
			strw.WriteLine("Type: TotalScanDriver /Interrupt <batchmanager> <process operation id> <interrupt string>");
			strw.WriteLine("to send an interrupt message to a running TotalScanDriver process operation.");
			strw.WriteLine("SUPPORTED INTERRUPTS:");
			strw.WriteLine("PlateDamagedCode <code> - instructs TotalScanDriver to use the specified code to mark the plate as damaged. The plate must be specified by PlateDamaged. If it's missing, the current plate number is assumed.");
			strw.WriteLine("PlateDamaged <number> - instructs TotalScanDriver to mark the specified plate as damaged. If it's missing, the current plate number is assumed.");
			strw.WriteLine("GoBackToPlateN <number> - instructs TotalScanDriver to go back to the specified plate, keeping intercalibration info.");
			strw.WriteLine("GoBackToPlateNCancelCalibrations <number> - instructs TotalScanDriver to go back to the specified plate, cancelling intercalibration info obtained by daughter operations of the current one. Intercalibrations obtained by previous operations will not be cancelled.");
			strw.WriteLine("Volumes <number> - used to provide volumes on startup when the volume source is set to Interrupt in the ProgramSettings. The number sets the number of expected volumes. Volumes are 7-tuples such as VOLUME MINX MAXX MINY MAXY MINPLATE MAXPLATE separated by ';'. The first volume must be preceeded by ';'. No ';' is to be put at the end of the volume string. The volume string may contain any spacers including newlines.");
			strw.WriteLine("Type: TotalScanDriver /EasyInterrupt for a graphical user interface to send interrupts.");			
			strw.WriteLine("--------------");
			strw.WriteLine("The program settings should have the following structure:");
			TotalScanSettings sbset = new TotalScanSettings();
			sbset.IntercalibrationConfigId = 80088238;
			sbset.AreaScanConfigId = 80088275;
			sbset.Direction = ScanDirection.Upstream;
			sbset.VolumeCreationMode = new VolumeCreation();
			sbset.VolumeCreationMode.DownstreamPlates = 8;
			sbset.VolumeCreationMode.UpstreamPlates = 1;
			sbset.VolumeCreationMode.Source = TotalScanDriver.InputSource.ScanbackPathFixedPrimarySlope;
			sbset.VolumeCreationMode.WidthFormula = "5000";
			sbset.VolumeCreationMode.PrimarySlope.X = 0.05;
			sbset.VolumeCreationMode.PrimarySlope.Y = -0.01;
			new System.Xml.Serialization.XmlSerializer(typeof(TotalScanSettings)).Serialize(strw, sbset);
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
            conn.Close();
        }

		private static SySal.DAQSystem.Drivers.HostEnv HE = null;

		private static SySal.OperaDb.OperaDbConnection Conn;

        private static SySal.OperaDb.OperaDbConnection CVConn;

		private static TotalScanSettings ProgSettings;

		private static SySal.DAQSystem.Drivers.VolumeOperationInfo StartupInfo;

		private static SySal.DAQSystem.Drivers.TaskProgressInfo ProgressInfo = new SySal.DAQSystem.Drivers.TaskProgressInfo();

		private static System.Threading.ManualResetEvent VolumeInterruptEvent = new System.Threading.ManualResetEvent(false);

		private static bool VolumeCreationDone;

		private static SySal.DAQSystem.Drivers.BoxDesc [] BoxesReceived = null;

		private static bool IntercalibrationDone;

		private static bool ScanDone;

		private static bool WaitForVolumes = false;

		private static int Plate;

		private static System.Threading.Thread ThisThread = null;

		private static long WaitingOnId;

		private static long CalibrationId;

		private static bool ReloadStatus = false;

		private static int MinPlate, MaxPlate;

		private static System.Drawing.Bitmap gIm = null;
		
		private static System.Drawing.Graphics gMon = null;

		private static NumericalTools.Plot gPlot = null;		

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
                        CreateVolumes();
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

		private static string ReplaceStrings(string s)
		{
			string ns = (string)s.Clone();
			ns = ns.Replace("%EXEREP%", StartupInfo.ExeRepository);
			ns = ns.Replace("%SCRATCH%", StartupInfo.ScratchDir);
			return ns;
		}

		private static void UpdateProgress()
		{
			ProgressInfo.CustomInfo = "\t\t[Plate] " + Plate.ToString() + " [/Plate]\r\n\t\t[WaitingOnId] " + WaitingOnId.ToString() + " [/WaitingOnId]\r\n\t\t[VolumeCreationDone] " + VolumeCreationDone +" [/VolumeCreationDone]\r\n\t\t[IntercalibrationDone] " + IntercalibrationDone + " [/IntercalibrationDone]\r\n\t\t[ScanDone] " + ScanDone + " [/ScanDone]\r\n\t\t[IgnoreCalibrationList]\r\n";
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
			try
			{
				w = new System.IO.StreamWriter(StartupInfo.ScratchDir + "\\" + StartupInfo.ProcessOperationId + "_progress.htm");
				w.WriteLine(
					"<html><head>" + (((ProgSettings.Direction == ScanDirection.Upstream && Plate < MaxPlate) || (ProgSettings.Direction == ScanDirection.Downstream && Plate > MinPlate)) ? "<meta http-equiv=\"REFRESH\" content=\"60\">" : "") + "<title>TotalScanDriver Monitor</title></head><body>\r\n" +
					"<div align=center><p><font face=\"Arial, Helvetica\" size=4 color=4444ff>TotalScanDriver Brick #" + StartupInfo.BrickId + "<br>Operation ID = " + StartupInfo.ProcessOperationId + "</font><hr></p></div>\r\n" +
					((VolumeCreationDone == false && ProgSettings.VolumeCreationMode.Source == InputSource.Interrupt) ? 
					"<div align=center><p><font face = \"Arial, Helvetica\" size=6 color=FF3333><b>Waiting for volume list from interrupt</b></font></p></div>\r\n" :
					"<div align=center><p><font face = \"Arial, Helvetica\" size=2 color=0000cc>Taking data</font></p></div>\r\n"
					) +
					"</body></html>"
					);
				w.Flush();
				w.Close();
			}
			catch (Exception) 
			{
				if (w != null) w.Close();
			}
		}

        static string[] PlateReductionFilters = new string[]
        {
            "",
            " AND MOD(ID - _MINID_,2) = 0 ",
            " AND MOD(ID - _MINID_,4) = 0 ",
            " AND (MOD(ID - _MINID_,2) = 0 OR (ID - _MINID_ <= 2) OR (_MAXID_ - ID <= 2)) ",
            " AND (MOD(ID - _MINID_,4) = 0 OR (ID - _MINID_ <= 4) OR (_MAXID_ - ID <= 4)) "
        };

		static void CreateVolumes()
		{
            try
            {                
                CVConn.Open();                
                SySal.OperaDb.Schema.DB = CVConn;
                string _minid_ = new SySal.OperaDb.OperaDbCommand("SELECT MIN(ID) FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId, CVConn).ExecuteScalar().ToString();
                string _maxid_ = new SySal.OperaDb.OperaDbCommand("SELECT MAX(ID) FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId, CVConn).ExecuteScalar().ToString();                
                string platefilter = PlateReductionFilters[(int)ProgSettings.VolumeCreationMode.ReductionScheme].Replace("_MINID_", _minid_).Replace("_MAXID_", _maxid_);
                if (ProgSettings.VolumeCreationMode.Source == InputSource.Interrupt)
                {                    
                    UpdateProgress();
                    UpdatePlots();
                    do
                    {
                        VolumeInterruptEvent.WaitOne(System.Threading.Timeout.Infinite, false);
                    }
                    while (BoxesReceived == null);                    
                    SySal.OperaDb.OperaDbTransaction trans = null;
                    try
                    {                        
                        trans = CVConn.BeginTransaction();                        
                        SySal.OperaDb.Schema.TB_EVENTBRICKS the_brick = SySal.OperaDb.Schema.TB_EVENTBRICKS.SelectPrimaryKey(StartupInfo.BrickId, SySal.OperaDb.Schema.OrderBy.None);
                        SySal.BasicTypes.Rectangle bounds = new SySal.BasicTypes.Rectangle();
                        the_brick.Row = 0;
                        bounds.MinX = the_brick._MINX - SySal.OperaDb.Convert.ToDouble(the_brick._ZEROX);
                        bounds.MaxX = the_brick._MAXX - SySal.OperaDb.Convert.ToDouble(the_brick._ZEROX);
                        bounds.MinY = the_brick._MINY - SySal.OperaDb.Convert.ToDouble(the_brick._ZEROY);
                        bounds.MaxY = the_brick._MAXY - SySal.OperaDb.Convert.ToDouble(the_brick._ZEROY);
                        bounds.MinX += ProgSettings.VolumeCreationMode.BoundMargin;
                        bounds.MaxX -= ProgSettings.VolumeCreationMode.BoundMargin;
                        bounds.MinY += ProgSettings.VolumeCreationMode.BoundMargin;
                        bounds.MaxY -= ProgSettings.VolumeCreationMode.BoundMargin;
                        foreach (ExtBoxDesc vol in BoxesReceived)
                        {
                            if (vol.ExtentsOnBottom.MaxX < vol.ExtentsOnBottom.MinX || vol.ExtentsOnBottom.MaxY < vol.ExtentsOnBottom.MinY) throw new Exception("Bad volume geometry: " + vol.Series + " MinX " + vol.ExtentsOnBottom.MinX + " MaxX " + vol.ExtentsOnBottom.MaxX + " MinY " + vol.ExtentsOnBottom.MinY + " MaxY " + vol.ExtentsOnBottom.MaxY + " .");                            
                            SySal.OperaDb.Schema.TB_PLATES pl = SySal.OperaDb.Schema.TB_PLATES.SelectPrimaryKey(StartupInfo.BrickId, vol.RefPlate, SySal.OperaDb.Schema.OrderBy.None);
                            pl.Row = 0;                            
                            long idvol = SySal.OperaDb.Schema.TB_VOLUMES.Insert(StartupInfo.BrickId, vol.Series, StartupInfo.ProcessOperationId, 0);
                            SySal.OperaDb.Schema.TB_PLATES the_plates = SySal.OperaDb.Schema.TB_PLATES.SelectWhere("ID BETWEEN " + vol.TopPlate + " AND " + vol.BottomPlate + " AND ID_EVENTBRICK = " + StartupInfo.BrickId + platefilter, null);                            
                            int platerows;
                            for (platerows = 0; platerows < the_plates.Count; platerows++)
                            {
                                the_plates.Row = platerows;
                                SySal.BasicTypes.Rectangle rect = vol.ExtentsOnBottom;
                                rect.MinX += (the_plates._Z - pl._Z) * vol.Slope.X;
                                rect.MaxX += (the_plates._Z - pl._Z) * vol.Slope.X;
                                rect.MinY += (the_plates._Z - pl._Z) * vol.Slope.Y;
                                rect.MaxY += (the_plates._Z - pl._Z) * vol.Slope.Y;
                                if (ProgSettings.VolumeCreationMode.CheckBounds)
                                {
                                    if (rect.MinX < bounds.MinX)
                                    {
                                        rect.MaxX = bounds.MinX + (rect.MaxX - rect.MinX);
                                        rect.MinX = bounds.MinX;
                                    }
                                    if (rect.MaxX > bounds.MaxX)
                                    {
                                        rect.MinX = bounds.MaxX - (rect.MaxX - rect.MinX);
                                        rect.MaxX = bounds.MaxX;
                                    }
                                    if (rect.MinY < bounds.MinY)
                                    {
                                        rect.MaxY = bounds.MinY + (rect.MaxY - rect.MinY);
                                        rect.MinY = bounds.MinY;
                                    }
                                    if (rect.MaxY > bounds.MaxY)
                                    {
                                        rect.MinY = bounds.MaxY - (rect.MaxY - rect.MinY);
                                        rect.MaxY = bounds.MaxY;
                                    }
                                }
                                object _idzone = System.DBNull.Value;
                                object _damaged = System.DBNull.Value;
                                if (vol.ReuseVolumeId > 0)
                                {
                                    bool IncludePlate = (vol.PlatesChosen == null);
                                    if (vol.PlatesChosen != null)
                                        foreach (int iplc in vol.PlatesChosen)
                                            if (iplc == the_plates._ID)
                                            {
                                                IncludePlate = true;
                                                break;
                                            }
                                    if (IncludePlate)
                                    {                                        
                                        SySal.OperaDb.Schema.TB_VOLUME_SLICES rv = SySal.OperaDb.Schema.TB_VOLUME_SLICES.SelectPrimaryKey(StartupInfo.BrickId, vol.ReuseVolumeId, the_plates._ID, SySal.OperaDb.Schema.OrderBy.Ascending);                                        
                                        if (rv.Count == 1)
                                        {
                                            rv.Row = 0;
                                            _idzone = rv._ID_ZONE;
                                            _damaged = rv._DAMAGED;
                                        }
                                    }
                                }                                
                                SySal.OperaDb.Schema.TB_VOLUME_SLICES.Insert(StartupInfo.BrickId, idvol, the_plates._ID, rect.MinX, rect.MaxX, rect.MinY, rect.MaxY, _idzone, _damaged);
                                if (_idzone != System.DBNull.Value) HE.WriteLine("Reused " + idvol + " plate " + the_plates._ID);
                            }                            
                            SySal.OperaDb.Schema.TB_VOLUME_SLICES.Flush();                            
                        }                        
                        trans.Commit();                        
                        VolumeCreationDone = true;
                        UpdateProgress();
                        trans = null;
                    }
                    catch (Exception x)
                    {
                        if (trans != null) trans.Rollback();
                        throw x;
                    }
                }
                else if (ProgSettings.VolumeCreationMode.Source == InputSource.ScanbackPath || ProgSettings.VolumeCreationMode.Source == InputSource.ScanbackPathFixedPrimarySlope)
                {
                    NumericalTools.CStyleParsedFunction wf = new NumericalTools.CStyleParsedFunction(ProgSettings.VolumeCreationMode.WidthFormula, "WidthFormula");
                    foreach (string p in wf.ParameterList)
                        if (String.Compare(p, "DZ", true) != 0 && String.Compare(p, "DPLATE", true) != 0)
                            throw new Exception("The only allowed parameters for width formula in volume creation are DZ and DPLATE.");
                    NumericalTools.CStyleParsedFunction hf = null;
                    if (ProgSettings.VolumeCreationMode.HeightFormula == null || ProgSettings.VolumeCreationMode.HeightFormula.Trim().Length == 0) hf = wf;
                    else hf = new NumericalTools.CStyleParsedFunction(ProgSettings.VolumeCreationMode.HeightFormula, "HeightFormula");
                    foreach (string p in hf.ParameterList)
                        if (String.Compare(p, "DZ", true) != 0 && String.Compare(p, "DPLATE", true) != 0)
                            throw new Exception("The only allowed parameters for height formula in volume creation are DZ and DPLATE.");
                    System.Data.DataSet dsbp = new System.Data.DataSet();
                    new SySal.OperaDb.OperaDbDataAdapter(
                        "SELECT /*+INDEX_ASC (TB_B_SBPATHS_VOLUMES IX_SBPATHS_VOLUMES) LEADING (TB_B_SBPATHS_VOLUMES) INDEX_ASC (TB_SCANBACK_PREDICTIONS PK_SCANBACK_PREDICTIONS) INDEX (TB_SCANBACK_PATHS PK_SCANBACK_PATHS) */ TB_B_SBPATHS_VOLUMES.PATH, TB_B_SBPATHS_VOLUMES.ID_PLATE, TB_MIPBASETRACKS.POSX, TB_MIPBASETRACKS.POSY, TB_MIPBASETRACKS.SLOPEX, TB_MIPBASETRACKS.SLOPEY, TB_B_SBPATHS_VOLUMES.ID_SCANBACK_PROCOPID FROM " +
                        "((TB_B_SBPATHS_VOLUMES INNER JOIN TB_SCANBACK_PATHS ON (TB_B_SBPATHS_VOLUMES.PATH = TB_SCANBACK_PATHS.PATH AND TB_B_SBPATHS_VOLUMES.ID_SCANBACK_PROCOPID = TB_SCANBACK_PATHS.ID_PROCESSOPERATION)) " +
                        "INNER JOIN TB_SCANBACK_PREDICTIONS ON (TB_SCANBACK_PATHS.ID_EVENTBRICK = TB_SCANBACK_PREDICTIONS.ID_EVENTBRICK AND TB_SCANBACK_PATHS.ID = TB_SCANBACK_PREDICTIONS.ID_PATH AND TB_B_SBPATHS_VOLUMES.ID_PLATE = TB_SCANBACK_PREDICTIONS.ID_PLATE)) INNER JOIN TB_MIPBASETRACKS ON " +
                        "(TB_SCANBACK_PREDICTIONS.ID_EVENTBRICK = TB_MIPBASETRACKS.ID_EVENTBRICK AND TB_SCANBACK_PREDICTIONS.ID_ZONE = TB_MIPBASETRACKS.ID_ZONE AND TB_SCANBACK_PREDICTIONS.ID_CANDIDATE = TB_MIPBASETRACKS.ID) WHERE TB_B_SBPATHS_VOLUMES.ID_EVENTBRICK = " + StartupInfo.BrickId +
                        " AND VOLUME IS NULL AND ID_VOLUMESCAN_PROCOPID IS NULL", CVConn, null).Fill(dsbp);
                    SySal.OperaDb.OperaDbTransaction trans = null;
                    try
                    {
                        trans = CVConn.BeginTransaction();
                        SySal.OperaDb.Schema.TB_EVENTBRICKS the_brick = SySal.OperaDb.Schema.TB_EVENTBRICKS.SelectPrimaryKey(StartupInfo.BrickId, SySal.OperaDb.Schema.OrderBy.None);
                        SySal.BasicTypes.Rectangle bounds = new SySal.BasicTypes.Rectangle();
                        the_brick.Row = 0;
                        bounds.MinX = the_brick._MINX - SySal.OperaDb.Convert.ToDouble(the_brick._ZEROX);
                        bounds.MaxX = the_brick._MAXX - SySal.OperaDb.Convert.ToDouble(the_brick._ZEROX);
                        bounds.MinY = the_brick._MINY - SySal.OperaDb.Convert.ToDouble(the_brick._ZEROY);
                        bounds.MaxY = the_brick._MAXY - SySal.OperaDb.Convert.ToDouble(the_brick._ZEROY);
                        bounds.MinX += ProgSettings.VolumeCreationMode.BoundMargin;
                        bounds.MaxX -= ProgSettings.VolumeCreationMode.BoundMargin;
                        bounds.MinY += ProgSettings.VolumeCreationMode.BoundMargin;
                        bounds.MaxY -= ProgSettings.VolumeCreationMode.BoundMargin;
                        int volid;

                        SySal.OperaDb.Schema.DB = CVConn;

                        for (volid = 1; volid <= dsbp.Tables[0].Rows.Count; volid++)
                        {
                            long idvol = SySal.OperaDb.Schema.TB_VOLUMES.Insert(StartupInfo.BrickId, volid, StartupInfo.ProcessOperationId, 0);
                            SySal.OperaDb.Schema.PC_SET_SBPATH_VOLUME.Call(StartupInfo.BrickId, SySal.OperaDb.Convert.ToInt64(dsbp.Tables[0].Rows[volid - 1][6]), SySal.OperaDb.Convert.ToInt64(dsbp.Tables[0].Rows[volid - 1][0]), SySal.OperaDb.Convert.ToInt64(dsbp.Tables[0].Rows[volid - 1][1]), StartupInfo.ProcessOperationId, volid);

                            long baseplate = SySal.OperaDb.Convert.ToInt64(dsbp.Tables[0].Rows[volid - 1][1]);
                            /*
                            long minplate = Math.Max(baseplate - ProgSettings.VolumeCreationMode.DownstreamPlates, MinPlate);
                            long maxplate = Math.Min(baseplate + ProgSettings.VolumeCreationMode.UpstreamPlates, MaxPlate);
                             */
                            System.Data.DataSet dplates = new System.Data.DataSet();
                            new SySal.OperaDb.OperaDbDataAdapter("with plates as (select id, row_number() over (order by Z asc) as zord from tb_plates where id_eventbrick = " +
                                StartupInfo.BrickId + ") select id, zord - basezord as dplate from plates inner join (select id as baseid, zord as basezord from plates where id = " +
                                baseplate + ") on ((zord - basezord) between " + (-ProgSettings.VolumeCreationMode.UpstreamPlates) + " and " + ProgSettings.VolumeCreationMode.DownstreamPlates +
                                ") order by dplate asc", CVConn, trans).Fill(dplates);
                            double width, height, z, basez;
                            double cx, cy;
                            basez = SySal.OperaDb.Convert.ToDouble(new SySal.OperaDb.OperaDbCommand("SELECT Z FROM VW_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + baseplate, CVConn, trans).ExecuteScalar());
                            foreach (System.Data.DataRow dr in dplates.Tables[0].Rows)
                            {
                                int plate = SySal.OperaDb.Convert.ToInt32(dr[0]);
                                int dplate = SySal.OperaDb.Convert.ToInt32(dr[1]);
                                z = SySal.OperaDb.Convert.ToDouble(new SySal.OperaDb.OperaDbCommand("SELECT Z FROM VW_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + plate, CVConn, trans).ExecuteScalar());
                                foreach (string p in wf.ParameterList)
                                    if (String.Compare(p, "DZ", true) == 0) wf[p] = z - basez;
                                    else if (String.Compare(p, "DPLATE", true) == 0) wf[p] = (double)dplate;
                                    else throw new Exception("Code was not updated. - Program write-time error - ask programmer about width formula parameters.");
                                width = wf.Evaluate();
                                foreach (string p in hf.ParameterList)
                                    if (String.Compare(p, "DZ", true) == 0) hf[p] = z - basez;
                                    else if (String.Compare(p, "DPLATE", true) == 0) hf[p] = (double)dplate;
                                    else throw new Exception("Code was not updated. - Program write-time error - ask programmer about height formula parameters.");
                                height = hf.Evaluate();
                                cx = SySal.OperaDb.Convert.ToDouble(dsbp.Tables[0].Rows[volid - 1][2]) + (z - basez) * ((ProgSettings.VolumeCreationMode.Source == InputSource.ScanbackPathFixedPrimarySlope) ? ProgSettings.VolumeCreationMode.PrimarySlope.X : SySal.OperaDb.Convert.ToDouble(dsbp.Tables[0].Rows[volid - 1][4]));
                                cy = SySal.OperaDb.Convert.ToDouble(dsbp.Tables[0].Rows[volid - 1][3]) + (z - basez) * ((ProgSettings.VolumeCreationMode.Source == InputSource.ScanbackPathFixedPrimarySlope) ? ProgSettings.VolumeCreationMode.PrimarySlope.Y : SySal.OperaDb.Convert.ToDouble(dsbp.Tables[0].Rows[volid - 1][5]));
                                SySal.BasicTypes.Rectangle rect = new SySal.BasicTypes.Rectangle();
                                rect.MinX = cx - 0.5 * width;
                                rect.MaxX = cx + 0.5 * width;
                                rect.MinY = cy - 0.5 * height;
                                rect.MaxY = cy + 0.5 * height;
                                if (rect.MinX < bounds.MinX)
                                {
                                    rect.MaxX = bounds.MinX + (rect.MaxX - rect.MinX);
                                    rect.MinX = bounds.MinX;
                                }
                                if (rect.MaxX > bounds.MaxX)
                                {
                                    rect.MinX = bounds.MaxX - (rect.MaxX - rect.MinX);
                                    rect.MaxX = bounds.MaxX;
                                }
                                if (rect.MinY < bounds.MinY)
                                {
                                    rect.MaxY = bounds.MinY + (rect.MaxY - rect.MinY);
                                    rect.MinY = bounds.MinY;
                                }
                                if (rect.MaxY > bounds.MaxY)
                                {
                                    rect.MinY = bounds.MaxY - (rect.MaxY - rect.MinY);
                                    rect.MaxY = bounds.MaxY;
                                }
                                SySal.OperaDb.Schema.TB_VOLUME_SLICES.Insert(StartupInfo.BrickId, idvol, plate, rect.MinX, rect.MaxX, rect.MinY, rect.MaxY, System.DBNull.Value, System.DBNull.Value);
                            }
                        }
                        SySal.OperaDb.Schema.TB_VOLUME_SLICES.Flush();
                        trans.Commit();
                        VolumeCreationDone = true;
                        UpdateProgress();
                        trans = null;
                    }
                    catch (Exception x)
                    {
                        if (trans != null) trans.Rollback();
                        throw x;
                    }
                }
                else throw new Exception("Unsupported volume creation mode!");
            }
            finally
            {                
                CVConn.Close();                
            }
		}

		private static void Execute()
		{
			ThisThread = System.Threading.Thread.CurrentThread;
			gIm = new System.Drawing.Bitmap(500, 375);
			gMon = System.Drawing.Graphics.FromImage(gIm);
			gPlot = new NumericalTools.Plot();

			StartupInfo = (SySal.DAQSystem.Drivers.VolumeOperationInfo)HE.StartupInfo;
            CVConn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
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
            MaxPlate = Convert.ToInt32(ds.Tables[0].Rows[1][0]);

			System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(TotalScanSettings));
			ProgSettings = (TotalScanSettings)xmls.Deserialize(new System.IO.StringReader(HE.ProgramSettings));
			xmls = null;

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
					xmldoc.LoadXml("<CustomInfo>"+ProgressInfo.CustomInfo.Replace('[','<').Replace(']','>')+"</CustomInfo>");
					System.Xml.XmlNode xmln = xmldoc.FirstChild;
					Plate = Convert.ToInt32(xmln["Plate"].InnerText);
					WaitingOnId = Convert.ToInt64(xmln["WaitingOnId"].InnerText);
					VolumeCreationDone = Convert.ToBoolean(xmln["VolumeCreationDone"].InnerText);
					IntercalibrationDone = Convert.ToBoolean(xmln["IntercalibrationDone"].InnerText);
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
				catch (Exception x)
				{
					HE.WriteLine("Restarting failed - proceeding to re-initialize process.");
					ProgressInfo = HE.ProgressInfo;
					VolumeCreationDone = false;
					IntercalibrationDone = false;
					ScanDone = false;
					Plate = (ProgSettings.Direction == ScanDirection.Upstream) ? MaxPlate : MinPlate;
					ProgressInfo.Progress = 0.0;
					ProgressInfo.StartTime = System.DateTime.Now;
					ProgressInfo.FinishTime = ProgressInfo.StartTime.AddYears(1);
				}
			}
			else
			{
				VolumeCreationDone = false;
				IntercalibrationDone = false;
				ScanDone = false;
				WaitingOnId = 0;
				Plate = (ProgSettings.Direction == ScanDirection.Upstream) ? MaxPlate : MinPlate;
				ProgressInfo = HE.ProgressInfo;
				ProgressInfo.Complete = false;
				ProgressInfo.ExitException = null;
				ProgressInfo.Progress = 0.0;
				ProgressInfo.StartTime = System.DateTime.Now;
				ProgressInfo.FinishTime = ProgressInfo.StartTime.AddYears(1);
			}
			if (VolumeCreationDone == false &&
				ProgSettings.VolumeCreationMode.Source == InputSource.Interrupt &&
				(ProgSettings.Direction == ScanDirection.Upstream && Plate == MaxPlate) ||
				(ProgSettings.Direction == ScanDirection.Downstream && Plate == MinPlate)) 
				WaitForVolumes = true;

            Conn.Close();
            Exe e = new Exe();
            HE.InterruptNotifier = e;
            AppDomain.CurrentDomain.SetData(SySal.DAQSystem.Drivers.HostEnv.WebAccessString, e);
            
            if (!VolumeCreationDone) CreateVolumes();
			UpdateProgress();
			UpdatePlots();
            
			WorkerThread.Start();
			HE.WriteLine("Entering TotalScan cycle.");            
            while (
#if false
			(ProgSettings.Direction == ScanDirection.Upstream && Plate <= MaxPlate) || (ProgSettings.Direction == ScanDirection.Downstream && Plate >= MinPlate))
#else
                Plate >= 0
#endif
            )
			{
				SySal.DAQSystem.Drivers.Status status;
				ReloadStatus = false;
				if (ReloadStatus) continue;
                Conn.Open();
                bool check = String.Compare(new SySal.OperaDb.OperaDbCommand("SELECT DAMAGED FROM VW_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + Plate, Conn, null).ExecuteScalar().ToString(), "N", true) == 0;
                Conn.Close();
				if (check)
				{
					// First step: Intercalibration
					if (IntercalibrationDone == false && ProgSettings.IntercalibrationConfigId != ProgSettings.AreaScanConfigId)
					{				
						CalibrationId = 0;
						if (WaitingOnId == 0)
						{
							object o = null;
							SySal.OperaDb.OperaDbConnection tempConn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
							tempConn.Open();
							o = new SySal.OperaDb.OperaDbCommand("SELECT CALIBRATION FROM VW_PLATES WHERE (ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + Plate + ")", tempConn, null).ExecuteScalar();
							tempConn.Close();
							if (ForceCalibrationList.Count > 0)
							{
								System.Data.DataSet dfc = new System.Data.DataSet();
								string fcstr = "";
								foreach (long ic in ForceCalibrationList)
									if (fcstr.Length > 0) fcstr += "," + ic.ToString();
									else fcstr = ic.ToString();
								new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_PROCESSOPERATION FROM TB_PLATE_CALIBRATIONS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PLATE = " + Plate + " AND ID_PROCESSOPERATION IN (" + fcstr + ")", tempConn, null).Fill(dfc);
								if (dfc.Tables[0].Rows.Count > 1)
								{
									fcstr = "";
									foreach (System.Data.DataRow dr in dfc.Tables[0].Rows)
										fcstr += "\r\n" + dr[0];
									throw new Exception("Ambiguity in calibration specification - found the following conflicting calibrations:" + fcstr);
								}
								else if (dfc.Tables[0].Rows.Count == 1) CalibrationId = SySal.OperaDb.Convert.ToInt64(dfc.Tables[0].Rows[0][0]);
							}

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
								intercalstartupinfo.ProgramSettingsId = ProgSettings.IntercalibrationConfigId;
								intercalstartupinfo.ProgressFile = "";
								intercalstartupinfo.RawDataPath = StartupInfo.RawDataPath;
								intercalstartupinfo.RecoverFromProgressFile = false;
								intercalstartupinfo.ScratchDir = StartupInfo.ScratchDir;
								intercalstartupinfo.Zones = new SySal.DAQSystem.Scanning.ZoneDesc[0];
								WaitingOnId = HE.Start(intercalstartupinfo);
								UpdateProgress();
							}
						}
						if (WaitingOnId != 0)
						{
							status = HE.Wait(WaitingOnId);
							lock(StartupInfo)
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
					else HE.WriteLine("Plate #" + Plate + " Intercalibration skipped because of program settings.");	
				
					// Second step: scanning
					lock(StartupInfo)
						if (ReloadStatus) 
							continue;
					if (ScanDone == false)
					{
						if (WaitingOnId == 0)
						{
                            Conn.Open();
							if (SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT /*+INDEX_ASC (TB_VOLUME_SLICES PK_VOLUME_SLICES) */ COUNT(*) FROM TB_VOLUMES INNER JOIN TB_VOLUME_SLICES ON (TB_VOLUMES.ID_EVENTBRICK = TB_VOLUME_SLICES.ID_EVENTBRICK AND TB_VOLUMES.ID = TB_VOLUME_SLICES.ID_VOLUME) WHERE TB_VOLUMES.ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PLATE = " + Plate + " AND ID_PROCESSOPERATION = " + StartupInfo.ProcessOperationId, Conn, null).ExecuteScalar()) > 0)
							{
								SySal.DAQSystem.Drivers.ScanningStartupInfo areascanstartupinfo = new SySal.DAQSystem.Drivers.ScanningStartupInfo();
								areascanstartupinfo.DBPassword = StartupInfo.DBPassword;
								areascanstartupinfo.DBServers = StartupInfo.DBServers;
								areascanstartupinfo.DBUserName = StartupInfo.DBUserName;
								areascanstartupinfo.ExeRepository = StartupInfo.ExeRepository;
								areascanstartupinfo.LinkedZonePath = StartupInfo.LinkedZonePath;
								areascanstartupinfo.MachineId = StartupInfo.MachineId;
								areascanstartupinfo.Plate = new SySal.DAQSystem.Scanning.MountPlateDesc();
								areascanstartupinfo.Plate.BrickId = StartupInfo.BrickId;
								areascanstartupinfo.Plate.PlateId = Plate;
                                areascanstartupinfo.Plate.MapInitString = (CalibrationId > 0)
                                    ? SySal.OperaDb.Scanning.Utilities.GetMapString(areascanstartupinfo.Plate.BrickId, areascanstartupinfo.Plate.PlateId, areascanstartupinfo.CalibrationId = CalibrationId, Conn, null)
                                    : SySal.OperaDb.Scanning.Utilities.GetMapString(areascanstartupinfo.Plate.BrickId, areascanstartupinfo.Plate.PlateId, false, SySal.OperaDb.Scanning.Utilities.CharToMarkType(Convert.ToChar(new SySal.OperaDb.OperaDbCommand("SELECT MARKSET FROM TB_PROGRAMSETTINGS WHERE ID = " + ProgSettings.AreaScanConfigId, Conn).ExecuteScalar().ToString().Trim())),
                                        out areascanstartupinfo.CalibrationId, Conn, null);
								areascanstartupinfo.Plate.TextDesc = "Brick #" + areascanstartupinfo.Plate.BrickId + " Plate #" + areascanstartupinfo.Plate.PlateId;
								areascanstartupinfo.ProcessOperationId = 0;
								areascanstartupinfo.ProgramSettingsId = ProgSettings.AreaScanConfigId;
								areascanstartupinfo.ProgressFile = "";
								areascanstartupinfo.RawDataPath = StartupInfo.RawDataPath;
								areascanstartupinfo.RecoverFromProgressFile = false;
								areascanstartupinfo.ScratchDir = StartupInfo.ScratchDir;
								areascanstartupinfo.Zones = new SySal.DAQSystem.Scanning.ZoneDesc[0];
								WaitingOnId = HE.Start(areascanstartupinfo);
								UpdateProgress();
								UpdatePlots();
							}
                            Conn.Close();
						}					
						status = HE.WaitForOpOrScanServer(WaitingOnId);
						lock(StartupInfo)
							if (ReloadStatus) 
								continue;
						if (status == SySal.DAQSystem.Drivers.Status.Failed)
						{
							WaitingOnId = 0;
							throw new Exception("Scanning failed on brick " + StartupInfo.BrickId + " , plate " + Plate + "!\r\nTotalScan interrupted.");
						}
						ScanDone = true;
						if (ProgSettings.IntercalibrationConfigId == ProgSettings.AreaScanConfigId) IntercalibrationDone = true;
						WaitingOnId = 0;
						UpdateProgress();	
						UpdatePlots();
					}
					HE.WriteLine("Plate #" + Plate + " Scanning OK");
				}
				else
				{
					HE.WriteLine("Plate #" + Plate + " scanning skipped because of plate damage condition; updating volume slices for this plate.");
					SySal.OperaDb.Schema.PC_EMPTY_VOLUMESLICES.Call(StartupInfo.BrickId, StartupInfo.ProcessOperationId, Plate);
					HE.WriteLine("Plate #" + Plate + " Skipped, moving to next plate.");
				}

                if (ProgSettings.Direction == ScanDirection.Upstream)
                {
                    Conn.Open();
                    object o = new SySal.OperaDb.OperaDbCommand("select id from (select /*+index_asc(tb_plates pk_plates) */ ID, row_number() over (order by z desc) as rnum from (select idp as id, zp as z from (select id_eventbrick as idb, id as idp, z as zp from tb_plates where id_eventbrick = "
                        + StartupInfo.BrickId + " and z < (select /*+index(tb_plates pk_plates) */ z from tb_plates where id_eventbrick = " + StartupInfo.BrickId + " and id = " + Plate + ") and id <> " + Plate + " and exists (select * from tb_volume_slices where (id_eventbrick, id_volume) in (select id_eventbrick, id from tb_volumes where id_eventbrick = " + StartupInfo.BrickId + " and id_processoperation = " + StartupInfo.ProcessOperationId + ") " +
                        " and id = id_plate and id_zone is null and damaged is null)) inner join vw_plates on (id_eventbrick = idb and idp = id and damaged = 'N'))) where rnum = 1",
                        Conn, null).ExecuteScalar();
                    Conn.Close();
                    if (o == System.DBNull.Value || o == null) Plate = -1;
                    else Plate = SySal.OperaDb.Convert.ToInt32(o);
                }
                else
                {
                    Conn.Open();
                    object o = new SySal.OperaDb.OperaDbCommand("select id from (select /*+index_asc(tb_plates pk_plates) */ ID, row_number() over (order by z asc) as rnum from (select idp as id, zp as z from (select id_eventbrick as idb, id as idp, z as zp from tb_plates where id_eventbrick = "
                        + StartupInfo.BrickId + " and z > (select /*+index(tb_plates pk_plates) */ z from tb_plates where id_eventbrick = " + StartupInfo.BrickId + " and id = " + Plate + ") and id <> " + Plate + " and exists (select * from tb_volume_slices where (id_eventbrick, id_volume) in (select id_eventbrick, id from tb_volumes where id_eventbrick = " + StartupInfo.BrickId + " and id_processoperation = " + StartupInfo.ProcessOperationId + ") " +
                        " and id = id_plate and id_zone is null and damaged is null)) inner join vw_plates on (id_eventbrick = idb and idp = id and damaged = 'N'))) where rnum = 1",
                        Conn, null).ExecuteScalar();
                    Conn.Close();
                    if (o == System.DBNull.Value || o == null) Plate = -1;
                    else Plate = SySal.OperaDb.Convert.ToInt32(o);
                }
                IntercalibrationDone = false;
				ScanDone = false;
				ProgressInfo.FinishTime = ProgressInfo.StartTime + System.TimeSpan.FromMilliseconds((System.DateTime.Now - ProgressInfo.StartTime).TotalMilliseconds * (1.0 - ProgressInfo.Progress));
				UpdateProgress();
				UpdatePlots();
			}
			lock(WorkQueue)
			{
				WorkQueue.Clear();
				WorkerThread.Interrupt();
			}
			WorkerThread.Join();
            System.Data.DataSet dsop = new System.Data.DataSet();
            Conn.Open();
            new SySal.OperaDb.OperaDbDataAdapter("SELECT ID FROM TB_PROC_OPERATIONS WHERE ID_PARENT_OPERATION = " + StartupInfo.ProcessOperationId + " AND SUCCESS = 'R'", Conn).Fill(dsop);
            Conn.Close();
            foreach (System.Data.DataRow drwop in dsop.Tables[0].Rows)
                HE.Wait(SySal.OperaDb.Convert.ToInt64(drwop[0]));
			ProgressInfo.Progress = 1.0;
			ProgressInfo.Complete = true;
			ProgressInfo.FinishTime = System.DateTime.Now;
			UpdateProgress();
			UpdatePlots();			
		}

		private static System.Text.RegularExpressions.Regex ForceCalEx = new System.Text.RegularExpressions.Regex(@"(\d+)\s+");

		private static System.Text.RegularExpressions.Regex IgnoreCalEx = new System.Text.RegularExpressions.Regex(@"(\d+)\s+");
		
		private static System.Text.RegularExpressions.Regex VolsEx = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\d+)");

		private static System.Text.RegularExpressions.Regex TwoEx = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)\s*");

		private static System.Text.RegularExpressions.Regex r_tok = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*");

        private static System.Text.RegularExpressions.Regex r_tok2 = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*");

        private static System.Text.RegularExpressions.Regex r_tok3 = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s*");

        private static System.Text.RegularExpressions.Regex r_tok4 = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\d+\s*\(\s*\d+[\s+\d+]*\s*\))\s*");

		#region IInterruptNotifier Members

        /// <summary>
        /// Called by the host BatchManager to notify the process of a new interrupt.
        /// </summary>
        /// <param name="nextint">the id of the next interrupt available in the queue.</param>
		public void NotifyInterrupt(Interrupt nextint)
		{
			lock(StartupInfo)
			{
                try
                {
                    Conn.Open();
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
                            System.Text.RegularExpressions.Match m = TwoEx.Match(line);
                            if (m.Success)
                            {
                                if (WaitForVolumes)
                                {
                                    m = VolsEx.Match(line);
                                    if (m.Success == false) continue;
                                    SySal.DAQSystem.Drivers.BoxDesc[] volsread = null;
                                    if (String.Compare(m.Groups[1].Value, "Volumes", true) == 0)
                                    {
                                        int ExpectedVolumes = 0;
                                        try
                                        {
                                            ExpectedVolumes = Convert.ToInt32(m.Groups[2].Value);
                                            volsread = new SySal.DAQSystem.Drivers.BoxDesc[ExpectedVolumes];
                                            int pr = 0;
                                            string[] prstr = line.Split(';');
                                            if (prstr.Length != ExpectedVolumes + 1) throw new Exception("Volume count does not match actual number of volumes.");
                                            for (pr = 0; pr < volsread.Length; pr++)
                                            {
                                                ExtBoxDesc p = new ExtBoxDesc();
                                                m = r_tok4.Match(prstr[pr + 1]);
                                                if (m.Success && m.Length == prstr[pr + 1].Length)
                                                {
                                                    p.Series = Convert.ToInt32(m.Groups[1].Value);
                                                    p.ExtentsOnBottom.MinX = Convert.ToDouble(m.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.ExtentsOnBottom.MaxX = Convert.ToDouble(m.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.ExtentsOnBottom.MinY = Convert.ToDouble(m.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.ExtentsOnBottom.MaxY = Convert.ToDouble(m.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.CenterOnBottom.X = 0.5 * (p.ExtentsOnBottom.MinX + p.ExtentsOnBottom.MaxX);
                                                    p.CenterOnBottom.Y = 0.5 * (p.ExtentsOnBottom.MinY + p.ExtentsOnBottom.MaxY);
                                                    p.Slope.X = Convert.ToDouble(m.Groups[9].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.Slope.Y = Convert.ToDouble(m.Groups[10].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.RefPlate = Convert.ToInt32(m.Groups[8].Value);
                                                    p.TopPlate = Convert.ToInt32(m.Groups[6].Value);
                                                    p.BottomPlate = Convert.ToInt32(m.Groups[7].Value);
                                                    string vspec = m.Groups[11].Value.Replace(')', ' ').Replace('(', ' ').Replace('\t', ' ').Replace('\r', ' ').Replace('\n', ' ');
                                                    string nspec;
                                                    while ((nspec = vspec.Replace("  ", " ").Trim()).Length != vspec.Length) vspec = nspec;
                                                    string[] vspl = nspec.Split(' ');
                                                    p.ReuseVolumeId = Convert.ToInt64(vspl[0]);
                                                    p.PlatesChosen = new int[vspl.Length - 1];
                                                    int vp;
                                                    for (vp = 1; vp < vspl.Length; vp++)
                                                        p.PlatesChosen[vp - 1] = Convert.ToInt32(vspl[vp]);
                                                    volsread[pr] = p;
                                                    continue;
                                                }
                                                m = r_tok3.Match(prstr[pr + 1]);
                                                if (m.Success && m.Length == prstr[pr + 1].Length)
                                                {
                                                    p.Series = Convert.ToInt32(m.Groups[1].Value);
                                                    p.ExtentsOnBottom.MinX = Convert.ToDouble(m.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.ExtentsOnBottom.MaxX = Convert.ToDouble(m.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.ExtentsOnBottom.MinY = Convert.ToDouble(m.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.ExtentsOnBottom.MaxY = Convert.ToDouble(m.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.CenterOnBottom.X = 0.5 * (p.ExtentsOnBottom.MinX + p.ExtentsOnBottom.MaxX);
                                                    p.CenterOnBottom.Y = 0.5 * (p.ExtentsOnBottom.MinY + p.ExtentsOnBottom.MaxY);
                                                    p.Slope.X = Convert.ToDouble(m.Groups[9].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.Slope.Y = Convert.ToDouble(m.Groups[10].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.RefPlate = Convert.ToInt32(m.Groups[8].Value);
                                                    p.TopPlate = Convert.ToInt32(m.Groups[6].Value);
                                                    p.BottomPlate = Convert.ToInt32(m.Groups[7].Value);
                                                    p.ReuseVolumeId = Convert.ToInt64(m.Groups[11].Value);
                                                    volsread[pr] = p;
                                                    continue;
                                                }
                                                m = r_tok2.Match(prstr[pr + 1]);
                                                if (m.Success && m.Length == prstr[pr + 1].Length)
                                                {
                                                    p.Series = Convert.ToInt32(m.Groups[1].Value);
                                                    p.ExtentsOnBottom.MinX = Convert.ToDouble(m.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.ExtentsOnBottom.MaxX = Convert.ToDouble(m.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.ExtentsOnBottom.MinY = Convert.ToDouble(m.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.ExtentsOnBottom.MaxY = Convert.ToDouble(m.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.CenterOnBottom.X = 0.5 * (p.ExtentsOnBottom.MinX + p.ExtentsOnBottom.MaxX);
                                                    p.CenterOnBottom.Y = 0.5 * (p.ExtentsOnBottom.MinY + p.ExtentsOnBottom.MaxY);
                                                    p.Slope.X = Convert.ToDouble(m.Groups[9].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.Slope.Y = Convert.ToDouble(m.Groups[10].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                    p.RefPlate = Convert.ToInt32(m.Groups[8].Value);
                                                    p.TopPlate = Convert.ToInt32(m.Groups[6].Value);
                                                    p.BottomPlate = Convert.ToInt32(m.Groups[7].Value);
                                                    p.ReuseVolumeId = 0;
                                                    volsread[pr] = p;
                                                    continue;
                                                }
                                                m = r_tok.Match(prstr[pr + 1]);
                                                if (m.Success != true || m.Length != prstr[pr + 1].Length) throw new Exception("Wrong volume syntax at volume " + pr);
                                                p.Series = Convert.ToInt32(m.Groups[1].Value);
                                                p.ExtentsOnBottom.MinX = Convert.ToDouble(m.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                p.ExtentsOnBottom.MaxX = Convert.ToDouble(m.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                p.ExtentsOnBottom.MinY = Convert.ToDouble(m.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                p.ExtentsOnBottom.MaxY = Convert.ToDouble(m.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture);
                                                p.CenterOnBottom.X = 0.5 * (p.ExtentsOnBottom.MinX + p.ExtentsOnBottom.MaxX);
                                                p.CenterOnBottom.Y = 0.5 * (p.ExtentsOnBottom.MinY + p.ExtentsOnBottom.MaxY);
                                                p.Slope.X = p.Slope.Y = 0.0;
                                                p.TopPlate = Convert.ToInt32(m.Groups[6].Value);
                                                p.BottomPlate = Convert.ToInt32(m.Groups[7].Value);
                                                p.RefPlate = p.TopPlate;
                                                p.ReuseVolumeId = 0;
                                                volsread[pr] = p;
                                            }
                                        }
                                        catch (Exception x)
                                        {
                                            HE.WriteLine(x.ToString());
                                            volsread = null;
                                            continue;
                                        }
                                        BoxesReceived = volsread;
                                        WaitForVolumes = false;
                                        VolumeInterruptEvent.Set();
                                    }
                                }
                                else if (String.Compare(m.Groups[1].Value, "ForceCalibrations", true) == 0)
                                {
                                    try
                                    {
                                        int ExpectedCalibrations = System.Convert.ToInt32(m.Groups[2].Value);
                                        System.Text.RegularExpressions.MatchCollection ms = ForceCalEx.Matches(line, m.Length);
                                        if (ExpectedCalibrations != ms.Count) throw new Exception("Number of expected calibrations does not match the number of IDs found.");
                                        foreach (System.Text.RegularExpressions.Match mt in ms)
                                        {
                                            long ic = System.Convert.ToInt64(mt.Groups[1].Value);
                                            int pos = ForceCalibrationList.BinarySearch(ic);
                                            if (pos < 0) ForceCalibrationList.Insert(~pos, ic);
                                        }
                                    }
                                    catch (Exception) { }
                                }
                                else if (String.Compare(m.Groups[1].Value, "IgnoreCalibrations", true) == 0)
                                {
                                    try
                                    {
                                        int ExpectedCalibrations = System.Convert.ToInt32(m.Groups[2].Value);
                                        System.Text.RegularExpressions.MatchCollection ms = IgnoreCalEx.Matches(line, m.Length);
                                        if (ExpectedCalibrations != ms.Count) throw new Exception("Number of expected calibrations does not match the number of IDs found.");
                                        foreach (System.Text.RegularExpressions.Match mt in ms)
                                        {
                                            long ic = System.Convert.ToInt64(mt.Groups[1].Value);
                                            int pos = IgnoreCalibrationList.BinarySearch(ic);
                                            if (pos < 0) IgnoreCalibrationList.Insert(~pos, ic);
                                        }
                                    }
                                    catch (Exception) { }
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
                                            CancelCalibrations = false;
                                            AbortCurrent = true;
                                            specGoBackToPlateN = true;
                                            if (ProgSettings.Direction == ScanDirection.Upstream)
                                            {
                                                if (GoBackToPlateN > Plate)
                                                {
                                                    GoBackToPlateN = -1;
                                                    CancelCalibrations = false;
                                                    AbortCurrent = false;
                                                    specGoBackToPlateN = false;
                                                }
                                            }
                                            else
                                            {
                                                if (GoBackToPlateN < Plate)
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
                                            CancelCalibrations = true;
                                            AbortCurrent = true;
                                            specGoBackToPlateN = true;
                                            if (ProgSettings.Direction == ScanDirection.Upstream)
                                            {
                                                if (GoBackToPlateN >= Plate)
                                                {
                                                    GoBackToPlateN = -1;
                                                    CancelCalibrations = false;
                                                    AbortCurrent = false;
                                                    specGoBackToPlateN = false;
                                                }
                                            }
                                            else
                                            {
                                                if (GoBackToPlateN <= Plate)
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
                                        if (ProgSettings.Direction == ScanDirection.Upstream)
                                        {
                                            GoBackToPlateN = Plate - 1;
                                        }
                                        else
                                        {
                                            GoBackToPlateN = Plate + 1;
                                        }
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
                                            //new SySal.OperaDb.OperaDbCommand("CALL PC_SCANBACK_DELETE_PREDICTIONS(" + StartupInfo.BrickId + ", " + StartupInfo.ProcessOperationId + ", " + Plate + ")", Conn, null).ExecuteNonQuery();
                                            ScanDone = false;
                                        }
                                    }
                                    if (specPlateDamagedCode)
                                        if (SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM VW_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID = " + PlateDamaged, Conn, null).ExecuteScalar()) == 1)
                                            new SySal.OperaDb.OperaDbCommand("CALL PC_SET_PLATE_DAMAGED(" + StartupInfo.BrickId + ", " + PlateDamaged + ", " + StartupInfo.ProcessOperationId + ", '" + PlateDamagedCode + "')", Conn, null).ExecuteNonQuery();
                                    if (specGoBackToPlateN)
                                    {
                                        if (ProgSettings.Direction == ScanDirection.Upstream)
                                        {
                                            SySal.OperaDb.OperaDbTransaction Trans = Conn.BeginTransaction();
                                            try
                                            {
                                                System.Data.DataSet dspl = new System.Data.DataSet();
                                                new SySal.OperaDb.OperaDbDataAdapter("SELECT /*+INDEX_ASC(TB_PLATES PK_PLATES) */ ID FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID >= " + GoBackToPlateN + " ORDER BY ID DESC", Conn, null).Fill(dspl);
                                                int newplate = GoBackToPlateN;
                                                foreach (System.Data.DataRow drpl in dspl.Tables[0].Rows)
                                                {
                                                    SySal.OperaDb.Schema.PC_EMPTY_VOLUMESLICES.Call(StartupInfo.BrickId, StartupInfo.ProcessOperationId, newplate = Convert.ToInt32(drpl[0]));
                                                    if (CancelCalibrations)
                                                    {
                                                        System.Data.DataSet dsi = new System.Data.DataSet();
                                                        new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_PROCESSOPERATION FROM TB_PLATE_CALIBRATIONS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PLATE = " + newplate, Conn, null).Fill(dsi);
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
                                            }
                                            catch (Exception x)
                                            {
                                                HE.WriteLine("Error during SQL transaction: " + x.Message);
                                                Trans.Rollback();
                                            }
                                        }
                                        else
                                        {
                                            SySal.OperaDb.OperaDbTransaction Trans = Conn.BeginTransaction();
                                            try
                                            {
                                                System.Data.DataSet dspl = new System.Data.DataSet();
                                                new SySal.OperaDb.OperaDbDataAdapter("SELECT /*+INDEX_ASC(TB_PLATES PK_PLATES) */ ID FROM TB_PLATES WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID <= " + GoBackToPlateN + " ORDER BY ID ASC", Conn, null).Fill(dspl);
                                                int newplate = GoBackToPlateN;
                                                foreach (System.Data.DataRow drpl in dspl.Tables[0].Rows)
                                                {
                                                    SySal.OperaDb.Schema.PC_EMPTY_VOLUMESLICES.Call(StartupInfo.BrickId, StartupInfo.ProcessOperationId, newplate = Convert.ToInt32(drpl[0]));
                                                    if (CancelCalibrations)
                                                    {
                                                        System.Data.DataSet dsi = new System.Data.DataSet();
                                                        new SySal.OperaDb.OperaDbDataAdapter("SELECT ID_PROCESSOPERATION FROM TB_PLATE_CALIBRATIONS WHERE ID_EVENTBRICK = " + StartupInfo.BrickId + " AND ID_PLATE = " + newplate, Conn, null).Fill(dsi);
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
                                            }
                                            catch (Exception x)
                                            {
                                                HE.WriteLine("Error during SQL transaction: " + x.Message);
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
                    HE.LastProcessedInterruptId = nextint.Id;
                }
                catch (Exception x)
                {
                    HE.WriteLine("Error processing interrupt: " + x.Message);
                }
                finally
                {
                    Conn.Close();
                }
				HE.LastProcessedInterruptId = nextint.Id;
			}
		}

		#endregion


        #region IWebApplication Members

        public string ApplicationName
        {
            get { return "SimpleScanback3Driver"; }
        }

        /// <summary>
        /// Processes HTTP GET method calls.
        /// </summary>
        /// <param name="sess">the user session.</param>
        /// <param name="page">the page requested (ignored).</param>
        /// <param name="queryget">commands sent to the page.</param>
        /// <returns>an HTML string with the page to be shown.</returns>
        public SySal.Web.ChunkedResponse HttpGet(SySal.Web.Session sess, string page, params string[] queryget)
        {
            return HttpPost(sess, page, queryget);
        }

        const string GoBackBtn = "gbk";
        const string GoBackCancelCalibsBtn = "gbkcc";
        const string GoBackPlateText = "gbkt";
        const string PlateDamagedBtn = "pdmg";
        const string PlateDamagedText = "pdmgt";
        const string PlateDamagedCodeText = "pdmgct";
        const string LoadPredictionsBtn = "lpred";
        const string LoadPredictionsText = "lpredt";

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
                Interrupt i = new Interrupt();
                i.Id = 0;
                foreach (string s in postfields)
                {
                    if (s.StartsWith(GoBackBtn + "="))
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
                    i.Data = "Volumes " + preds.Length;
                    foreach (string p in preds)
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
                "    <title>TotalScanDriver - " + StartupInfo.BrickId + "/" + StartupInfo.ProcessOperationId + "</title>\r\n" +
                "    <style type=\"text/css\">\r\n" +
                "    th { font-family: Arial,Helvetica; font-size: 12; color: white; background-color: teal; text-align: center; font-weight: bold }\r\n" +
                "    td { font-family: Arial,Helvetica; font-size: 12; color: navy; background-color: white; text-align: right; font-weight: normal }\r\n" +
                "    p {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: left; font-weight: normal }\r\n" +
                "    div {font-family: Arial,Helvetica; font-size: 14; color: black; background-color: white; text-align: center; font-weight: normal }\r\n" +
                "    </style>\r\n" +
                "</head>\r\n" +
                "<body>\r\n" +
                " <div>TotalScanDriver = " + StartupInfo.ProcessOperationId + "<br>Brick = " + StartupInfo.BrickId + "<br>Plate = " + Plate + "<br>Direction = " + ProgSettings.Direction + "</div>\r\n<hr>\r\n" +
                ((xctext != null) ? "<div>Interrupt Error:<br><font color=\"red\">" + SySal.Web.WebServer.HtmlFormat(xctext) + "<font></div>\r\n" : "") +
                " <form action=\"" + page + "\" method=\"post\" enctype=\"application/x-www-form-urlencoded\">\r\n" +
                "  <div>\r\n" +
                "   <input id=\"" + GoBackBtn + "\" name=\"" + GoBackBtn + "\" type=\"submit\" value=\"Go Back to Plate\"/>&nbsp;<input id=\"" + GoBackCancelCalibsBtn + "\" name=\"" + GoBackCancelCalibsBtn + "\" type=\"submit\" value=\"Go Back to Plate and Cancel Calibrations\"/>&nbsp;<input id=\"" + GoBackPlateText + "\" maxlength=\"3\" name=\"" + GoBackPlateText + "\" size=\"3\" type=\"text\" />\r\n" +
                "   <input id=\"" + PlateDamagedBtn + "\" name=\"" + PlateDamagedBtn + "\" type=\"submit\" value=\"Mark Plate Damaged\"/>&nbsp;<input id=\"" + PlateDamagedText + "\" maxlength=\"3\" name=\"" + PlateDamagedText + "\" size=\"3\" type=\"text\" value=\"" + Plate + "\" />&nbsp;<input id=\"" + PlateDamagedCodeText + "\" maxlength=\"3\" name=\"" + PlateDamagedCodeText + "\" size=\"3\" type=\"text\" value=\"N\" />\r\n" +
                "   <input id=\"" + LoadPredictionsBtn + "\" name=\"" + LoadPredictionsBtn + "\" type=\"submit\" value=\"Load Predictions\"/>&nbsp;<textarea id=\"" + LoadPredictionsText + "\" name=\"" + LoadPredictionsText + "\" size=\"3\" rows=\"4\" cols=\"20\" /></textarea>\r\n" +
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
