using System;
using System.Xml;
using System.Xml.Serialization;
using SySal.Processing.AlphaOmegaReconstruction;

namespace SySal.Executables.BatchReconstruct
{
	/// <summary>
	/// Batch reconstruct configuration.
	/// </summary>
	[Serializable]
	[XmlType("BatchReconstruct.Config")]
	public class Config
	{
		public Configuration ReconstructorConfig;
	}

	/// <summary>
	/// Zone specifier that defines the source and placement of a LinkedZone.
	/// </summary>
	[Serializable]
	public class Zone
	{
		/// <summary>
		/// Opera persistence path for the LinkedZone.
		/// </summary>
		public string Source;
		/// <summary>
		/// Sheet identifier.
		/// </summary>
		public long SheetId;
		/// <summary>
		/// Initial Z position of the LinkedZone in the volume.
		/// </summary>
        public double Z;
        /// <summary>
        /// Path to the file with the Ids of tracks to be ignored in alignment. This file is optional and can be <c>null</c> if no track is to be ignored.
        /// If not <c>null</c>, the file must contain the word <c>Index</c> on the first line, and, on each following line, a zero-based index of a track to be ignored for alignment computation.
        /// </summary>
        public string AlignmentIgnoreListPath;
	}
	/// <summary>
	/// Input for BatchReconstruct.
	/// </summary>
	[Serializable]
	public class Input
	{
		/// <summary>
		/// List of zones.
		/// </summary>
		public Zone [] Zones;
	}

	/// <summary>
	/// BatchReconstruct - performs volume reconstruction using LinkedZones from TLG files or OPERA DB tables.
	/// </summary>
	/// <remarks>
	/// <para>
	/// BatchReconstruct uses AlphaOmegaReconstructor. See <see cref="SySal.Processing.AlphaOmegaReconstruction.AlphaOmegaReconstructor"/> for more details about the meaning of the fields of the configuration file.
	/// </para>
	/// <para>
	/// Usage modes:
	/// <list type="bullet">
	/// <item><term><c>BatchReconstruct.exe &lt;XML list file&gt; &lt;output Opera persistence path&gt; &lt;XML config Opera persistence path&gt;</c></term></item>
	/// <item><term><c>BatchReconstruct.exe &lt;DB volume&gt; &lt;output Opera persistence path&gt; &lt;XML config Opera persistence path&gt;</c></term></item>
	/// <item><term><c>BatchReconstruct.exe &lt;input OPERA persistence path&gt; &lt;output Opera persistence path&gt; &lt;XML config Opera persistence path&gt;</c></term></item>
	/// </list>
	/// Notice: full volumes are reprocessed for topological reconstruction only.
	/// </para>
	/// <para>
	/// Full reconstruction example:
	/// <example>
	/// <code>
	/// &lt;Input&gt;
	///  &lt;Zones&gt;
	///   &lt;Zone&gt;
	///    &lt;Source&gt;\\myserver.mydomain\myshare\plate_08.tlg&lt;/Source&gt;
	///    &lt;SheetId&gt;8&lt;/SheetId&gt;
	///    &lt;Z&gt;0&lt;/Z&gt;
    ///    &lt;AlignmentIgnoreListPath&gt;\\myserver.mydomain\myshare\plate_08.tlg&lt;/AlignmentIgnoreListPath&gt;
	///   &lt;/Zone&gt;
	///   &lt;Zone&gt;
	///    &lt;Source&gt;\\myserver.mydomain\myshare\plate_09.tlg&lt;/Source&gt;
	///    &lt;SheetId&gt;9&lt;/SheetId&gt;
	///    &lt;Z&gt;-1300&lt;/Z&gt;
	///   &lt;/Zone&gt;
	///   &lt;Zone&gt;
	///    &lt;Source&gt;\\myserver.mydomain\myshare\plate_10.tlg&lt;/Source&gt;
	///    &lt;SheetId&gt;10&lt;/SheetId&gt;
	///    &lt;Z&gt;-2600&lt;/Z&gt;
	///   &lt;/Zone&gt;
	///  &lt;/Zones&gt;
	/// &lt;/Input&gt;
	/// </code>
	/// </example>
	/// </para>
 	/// <para>
	/// DB volume example:
	/// <example><c>db:\8\17723900.vol</c> The first number is the brick number, the second is the id_volume to be analyzed.</example>
	/// </para>
	/// <para>
	/// XML config file syntax:
	/// <code>
	/// &lt;BatchReconstruct.Config&gt;
	///  &lt;ReconstructorConfig&gt;
	///   &lt;Name&gt;Default AlphaOmega Configuration&lt;/Name&gt;
	///   &lt;TopologyV&gt;true&lt;/TopologyV&gt;
	///   &lt;TopologyKink&gt;true&lt;/TopologyKink&gt;
	///   &lt;TopologyX&gt;false&lt;/TopologyX&gt;
	///   &lt;TopologyY&gt;false&lt;/TopologyY&gt;
	///   &lt;TopologyLambda&gt;false&lt;/TopologyLambda&gt;
	///   &lt;MinVertexTracksSegments&gt;3&lt;/MinVertexTracksSegments&gt;
	///   &lt;Initial_D_Pos&gt;40&lt;/Initial_D_Pos&gt;
	///   &lt;Initial_D_Slope&gt;0.04&lt;/Initial_D_Slope&gt;
	///   &lt;MaxIters&gt;5&lt;/MaxIters&gt;
	///   &lt;D_PosIncrement&gt;20&lt;/D_PosIncrement&gt;
	///   &lt;D_SlopeIncrement&gt;0.025&lt;/D_SlopeIncrement&gt;
	///   &lt;D_Pos&gt;30&lt;/D_Pos&gt;
	///   &lt;D_Slope&gt;0.03&lt;/D_Slope&gt;
	///   &lt;LocalityCellSize&gt;250&lt;/LocalityCellSize&gt;
	///   &lt;AlignBeamSlope&gt;
	///   &lt;X&gt;0&lt;/X&gt;
	///   &lt;Y&gt;0&lt;/Y&gt;
	///   &lt;/AlignBeamSlope&gt;	
	///   &lt;AlignBeamWidth&gt;1&lt;/AlignBeamWidth&gt;
	///   &lt;FreezeZ&gt;false&lt;/FreezeZ&gt;
	///   &lt;CorrectSlopesAlign&gt;false&lt;/CorrectSlopesAlign&gt;
	///   &lt;AlignOnLinked&gt;false&lt;/AlignOnLinked&gt;
	///   &lt;MaxMissingSegments&gt;1&lt;/MaxMissingSegments&gt;
	///   &lt;PrescanMode&gt;Rototranslation&lt;/PrescanMode&gt;
	///   &lt;LeverArm&gt;1500&lt;/LeverArm&gt;
	///   &lt;ZoneWidth&gt;2500&lt;/ZoneWidth&gt;
	///   &lt;Extents&gt;1000&lt;/Extents&gt;
	///   &lt;RiskFactor&gt;0.01&lt;/RiskFactor&gt;
	///   &lt;SlopesCellSize&gt;
	///   &lt;X&gt;0.05&lt;/X&gt;
	///   &lt;Y&gt;0.05&lt;/Y&gt;
	///   &lt;/SlopesCellSize&gt;
	///   &lt;MaximumShift&gt;
	///   &lt;X&gt;1000&lt;/X&gt;
	///   &lt;Y&gt;1000&lt;/Y&gt;
	///   &lt;/MaximumShift&gt;
	///   &lt;CrossTolerance&gt;10&lt;/CrossTolerance&gt;
	///   &lt;MaximumZ&gt;6000&lt;/MaximumZ&gt;
	///   &lt;MinimumZ&gt;-6000&lt;/MinimumZ&gt;
	///   &lt;StartingClusterToleranceLong&gt;300&lt;/StartingClusterToleranceLong&gt;
	///   &lt;MaximumClusterToleranceLong&gt;600&lt;/MaximumClusterToleranceLong&gt;
	///   &lt;StartingClusterToleranceTrans&gt;30&lt;/StartingClusterToleranceTrans&gt;
	///   &lt;MaximumClusterToleranceTrans&gt;60&lt;/MaximumClusterToleranceTrans&gt;
	///   &lt;MinimumTracksPairs&gt;20&lt;/MinimumTracksPairs&gt;
	///   &lt;MinimumSegmentsNumber&gt;5&lt;/MinimumSegmentsNumber&gt;
	///   &lt;UpdateTransformations&gt;true&lt;/UpdateTransformations&gt;
	///   &lt;Matrix&gt;1.8&lt;/Matrix&gt;
	///   &lt;XCellSize&gt;250&lt;/XCellSize&gt;
	///   &lt;YCellSize&gt;250&lt;/YCellSize&gt;
	///   &lt;ZCellSize&gt;1300&lt;/ZCellSize&gt;
	///   &lt;UseCells&gt;true&lt;/UseCells&gt;
	///   &lt;KalmanFilter&gt;false&lt;/KalmanFilter&gt;
	///   &lt;FittingTracks&gt;3&lt;/FittingTracks&gt;
	///   &lt;MinKalman&gt;4&lt;/MinKalman&gt;
	///   &lt;MinimumCritical&gt;1&lt;/MinimumCritical&gt;
	///   &lt;KinkDetection&gt;false&lt;/KinkDetection&gt;
	///   &lt;KinkMinimumSegments&gt;6&lt;/KinkMinimumSegments&gt;
	///   &lt;KinkMinimumDeltaS&gt;0.02&lt;/KinkMinimumDeltaS&gt;
	///   &lt;KinkFactor&gt;1.1&lt;/KinkFactor&gt;
	///   &lt;FilterThreshold&gt;500&lt;/FilterThreshold&gt;
	///   &lt;FilterLength&gt;10&lt;/FilterLength&gt;
	///   &lt;VtxAlgorithm&gt;None&lt;/VtxAlgorithm&gt;
	///   &lt;GVtxMaxSlopeDivergence&gt;1.2&lt;/GVtxMaxSlopeDivergence&gt;
	///   &lt;GVtxRadius&gt;30&lt;/GVtxRadius&gt;
	///   &lt;GVtxMaxExt&gt;3900&lt;/GVtxMaxExt&gt;
	///   &lt;GVtxMinCount&gt;2&lt;/GVtxMinCount&gt;
	///   &lt;VtxFitWeightEnable&gt;true&lt;/VtxFitWeightEnable&gt;
	///   &lt;VtxFitWeightTol&gt;0.1&lt;/VtxFitWeightTol&gt;
	///   &lt;VtxFitWeightOptStepXY&gt;1&lt;/VtxFitWeightOptStepXY&gt;
	///   &lt;VtxFitWeightOptStepZ&gt;5&lt;/VtxFitWeightOptStepZ&gt;
	///  &lt;/ReconstructorConfig&gt;
	/// &lt;/BatchReconstruct.Config&gt;
	/// </code>
	/// </para>
	/// </remarks>
	public class Exe
	{
		private static System.DateTime LastTime = new System.DateTime(1,1,1);

		public static void Progress(double p)
		{
			System.DateTime d = System.DateTime.Now;
			if ((d - LastTime).TotalSeconds >= 2.0 || p >= 1.0) 
			{
				Console.WriteLine("Progress: {0}", p * 100.0);
				LastTime = d;
			}
		}

		public static void Report(string s)
		{
			Console.WriteLine(s);
		}
		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main(string[] args)
        {
            //
            // TODO: Add code to start application here
            //
            if (args.Length != 3)
            {
                System.Xml.Serialization.XmlSerializer xmls = null;
                Console.WriteLine("BatchReconstruct - performs volume reconstruction using LinkedZones from TLG files or OPERA DB tables.");
                Console.WriteLine("usage: batchreconstruct <XML list file> <output Opera persistence path> <XML config Opera persistence path>");
                Console.WriteLine("or:    batchreconstruct <DB volume> <output Opera persistence path> <XML config Opera persistence path>");
                Console.WriteLine("or:    batchreconstruct <input OPERA persistence path> <output Opera persistence path> <XML config Opera persistence path>");
                Console.WriteLine("Full volumes are reprocessed for topological reconstruction only.");
                Console.WriteLine("---------------- DB volume example: db:\\8\\17723900.vol");
                Console.WriteLine("First number is ID_EVENTBRICK, second is ID_VOLUME");
                Console.WriteLine("---------------- XML list file example (source = filesystem):");
                Input inputlist = new Input();
                inputlist.Zones = new Zone[3];
                inputlist.Zones[0] = new Zone();
                inputlist.Zones[0].SheetId = 8;
                inputlist.Zones[0].Source = @"\\myserver.mydomain\myshare\plate_08.tlg";
                inputlist.Zones[0].Z = 0.0;
                inputlist.Zones[0].AlignmentIgnoreListPath = @"\\myserver\mydomain\myshare\alignignore_plate_08.txt";
                inputlist.Zones[1] = new Zone();
                inputlist.Zones[1].SheetId = 9;
                inputlist.Zones[1].Source = @"\\myserver.mydomain\myshare\plate_09.tlg";
                inputlist.Zones[1].Z = -1300.0;
                inputlist.Zones[2] = new Zone();
                inputlist.Zones[2].SheetId = 10;
                inputlist.Zones[2].Source = @"\\myserver.mydomain\myshare\plate_10.tlg";
                inputlist.Zones[2].Z = -2600.0;
                xmls = new System.Xml.Serialization.XmlSerializer(typeof(BatchReconstruct.Input));
                xmls.Serialize(Console.Out, inputlist);
                Console.WriteLine();
                Console.WriteLine("---------------- XML list file example (source = OperaDB):");
                inputlist.Zones[0].Source = @"db:\1002323.tlg";
                inputlist.Zones[1].Source = @"db:\1006326.tlg";
                inputlist.Zones[2].Source = @"db:\1009724.tlg";
                xmls.Serialize(Console.Out, inputlist);
                Console.WriteLine();
                Console.WriteLine("---------------- XML config file syntax:");
                Console.WriteLine("XML configuration syntax:");
                BatchReconstruct.Config C = new BatchReconstruct.Config();
                C.ReconstructorConfig = (Configuration)new AlphaOmegaReconstructor().Config;
                xmls = new System.Xml.Serialization.XmlSerializer(typeof(BatchReconstruct.Config));
                xmls.Serialize(Console.Out, C);
                Console.WriteLine();
                return;
            }
#if !(DEBUG)
			try			
#endif
            {
                AlphaOmegaReconstructor R = new AlphaOmegaReconstructor();
                System.Xml.Serialization.XmlSerializer xmls = null;

                xmls = new System.Xml.Serialization.XmlSerializer(typeof(BatchReconstruct.Config));
                Config config = (Config)xmls.Deserialize(new System.IO.StringReader(((SySal.OperaDb.ComputingInfrastructure.ProgramSettings)SySal.OperaPersistence.Restore(args[2], typeof(SySal.OperaDb.ComputingInfrastructure.ProgramSettings))).Settings));
                R.Config = (SySal.Management.Configuration)config.ReconstructorConfig;
                R.Progress = new SySal.TotalScan.dProgress(Progress);
                R.Report = new SySal.TotalScan.dReport(Report);

                System.Text.RegularExpressions.Regex volrgx = new System.Text.RegularExpressions.Regex(@"db:\\(\d+)\\(\d+)\.vol");
                System.Text.RegularExpressions.Match mrgx = volrgx.Match(args[0].ToLower());
                Input inputlist = null;
                SySal.TotalScan.Volume OldVol = null;
                if (args[0].ToLower().EndsWith(".tsr"))
                {
                    OldVol = (SySal.TotalScan.Volume)SySal.OperaPersistence.Restore(args[0], typeof(SySal.TotalScan.Volume));
                }
                else
                {
                    if (mrgx.Success && mrgx.Length == args[0].Length)
                    {
                        SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
                        SySal.OperaDb.OperaDbConnection conn = new SySal.OperaDb.OperaDbConnection(cred.DBServer, cred.DBUserName, cred.DBPassword);
                        conn.Open();
                        System.Data.DataSet ds = new System.Data.DataSet();
                        new SySal.OperaDb.OperaDbDataAdapter("SELECT TB_VOLUME_SLICES.ID_PLATE, TB_VOLUME_SLICES.ID_ZONE, VW_PLATES.Z FROM TB_VOLUME_SLICES INNER JOIN VW_PLATES ON (TB_VOLUME_SLICES.ID_EVENTBRICK = VW_PLATES.ID_EVENTBRICK AND TB_VOLUME_SLICES.ID_PLATE = VW_PLATES.ID) WHERE TB_VOLUME_SLICES.DAMAGED = 'N' AND TB_VOLUME_SLICES.ID_EVENTBRICK = " + mrgx.Groups[1].Value + " AND TB_VOLUME_SLICES.ID_VOLUME = " + mrgx.Groups[2].Value + " ORDER BY VW_PLATES.Z DESC", conn, null).Fill(ds);
                        inputlist = new Input();
                        inputlist.Zones = new Zone[ds.Tables[0].Rows.Count];
                        int sli;
                        for (sli = 0; sli < ds.Tables[0].Rows.Count; sli++)
                        {
                            inputlist.Zones[sli] = new Zone();
                            inputlist.Zones[sli].SheetId = Convert.ToInt32(ds.Tables[0].Rows[sli][0]);
                            inputlist.Zones[sli].Source = "db:\\" + mrgx.Groups[1] + "\\" + ds.Tables[0].Rows[sli][1].ToString() + ".tlg";
                            inputlist.Zones[sli].Z = Convert.ToDouble(ds.Tables[0].Rows[sli][2]);
                        }
                        SySal.OperaPersistence.Connection = conn;
                        SySal.OperaPersistence.LinkedZoneDetailLevel = SySal.OperaDb.Scanning.LinkedZone.DetailLevel.BaseFull;
                    }
                    else
                    {
                        System.IO.StreamReader r = new System.IO.StreamReader(args[0]);
                        xmls = new System.Xml.Serialization.XmlSerializer(typeof(BatchReconstruct.Input));
                        inputlist = (Input)xmls.Deserialize(r);
                        r.Close();
                    }

                    int i, j, c;
                    for (i = 0; i < inputlist.Zones.Length; i++)
                        for (j = i + 1; j < inputlist.Zones.Length; j++)
                            if (inputlist.Zones[i].SheetId == inputlist.Zones[j].SheetId)
                            {
                                Console.WriteLine("Duplicate SheetId found. Sheets will be renumbered with the default sequence.");
                                for (j = 0; j < inputlist.Zones.Length; j++)
                                    inputlist.Zones[j].SheetId = j;
                                i = inputlist.Zones.Length;
                                break;
                            }
                    for (i = 0; i < inputlist.Zones.Length; i++)
                    {
                        SySal.Scanning.Plate.IO.OPERA.LinkedZone lz = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)SySal.OperaPersistence.Restore(inputlist.Zones[i].Source, typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone));
                        c = lz.Length;
                        SySal.TotalScan.Segment[] segs = new SySal.TotalScan.Segment[c];


                        double[] zcor = new double[c];
                        for (j = 0; j < c; j++)
                            zcor[j] = lz[j].Info.Intercept.Z;

                        double zmean = NumericalTools.Fitting.Average(zcor);
                        double dgap;
                        for (j = 0; j < c; j++)
                        {
                            SySal.Tracking.MIPEmulsionTrackInfo info = lz[j].Info;
                            segs[j] = new SySal.TotalScan.Segment(info, new SySal.TotalScan.BaseTrackIndex(j));
                            dgap = zmean - info.Intercept.Z;
                            info.Intercept.Z = zmean;
                            info.Intercept.X += info.Slope.X * dgap;
                            info.Intercept.Y += info.Slope.Y * dgap;
                            info.TopZ += dgap;
                            info.BottomZ += dgap;

                            info.Intercept.Z = inputlist.Zones[i].Z;
                            double tmptopz = info.TopZ;
                            double tmpbotz = info.BottomZ;
                            dgap = zmean - tmptopz;
                            info.TopZ = inputlist.Zones[i].Z - dgap;
                            info.BottomZ = inputlist.Zones[i].Z - (tmptopz - tmpbotz) - dgap;
                        }

                        SySal.BasicTypes.Vector refc = new SySal.BasicTypes.Vector();
                        refc.Z = inputlist.Zones[i].Z;
                        SySal.TotalScan.Layer tmpLayer = new SySal.TotalScan.Layer(i, /*System.Convert.ToInt64(mrgx.Groups[1].Value)*/0, (int)inputlist.Zones[i].SheetId, 0, refc);
                        tmpLayer.AddSegments(segs);
                        R.AddLayer(tmpLayer);
                        if (inputlist.Zones[i].AlignmentIgnoreListPath != null && inputlist.Zones[i].AlignmentIgnoreListPath.Trim().Length != 0)
                        {
                            if (inputlist.Zones[i].AlignmentIgnoreListPath.ToLower().EndsWith(".tlg"))
                            {
                                R.SetAlignmentIgnoreList(i, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIgnoreAlignment)SySal.OperaPersistence.Restore(inputlist.Zones[i].AlignmentIgnoreListPath, typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIgnoreAlignment))).Ids);
                            }
                            else
                            {
                                System.IO.StreamReader r = new System.IO.StreamReader(inputlist.Zones[i].AlignmentIgnoreListPath.Trim());
                                System.Collections.ArrayList tmpignorelist = new System.Collections.ArrayList();
                                string line;
                                while ((line = r.ReadLine()) != null)
                                    try
                                    {
                                        tmpignorelist.Add(System.Convert.ToInt32(line));
                                    }
                                    catch (Exception) { };
                                r.Close();
                                R.SetAlignmentIgnoreList(i, (int[])tmpignorelist.ToArray(typeof(int)));
                            }
                        }

                        Console.WriteLine("Loaded sheet {0} Id {1} Tracks {2}", i, inputlist.Zones[i].SheetId, c);
                        lz = null;
                    }
                }
                SySal.TotalScan.Volume V = (OldVol == null) ? R.Reconstruct() : R.RecomputeVertices(OldVol);
                Console.WriteLine("Result written to: " + SySal.OperaPersistence.Persist(args[1], V));
            }
#if !(DEBUG)			
			catch (Exception x)
			{
				Console.Error.WriteLine(x.ToString());
			}
#endif
        }
	}
}
