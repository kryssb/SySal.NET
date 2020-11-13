using System;

namespace SySal.Executables.OperaCopy
{
	/// <summary>
	/// OperaCopy - Command line tool for data transfer.
	/// </summary>
	/// <remarks>
	/// <para>OperaCopy is a utility built on top of the syntax of the Grand Unified File System, as described in <see cref="SySal.OperaPersistence">OperaPersistence</see>.</para>
	/// <para>The program copies data from a specified source to a specified destination, in several different formats.</para>
	/// <para>The typical syntax is: <c>OperaCopy.exe &lt;source&gt; &lt;destination&gt;</c></para>
	/// <para>
	/// Recognized formats:
	/// <list type="table">
	/// <listheader><term>Format</term><description>Description</description></listheader>
	/// <item><term>RWC</term><description>Raw Data Catalog. Only file-to-file copies are supported.</description></item>
	/// <item><term>RWD</term><description>Raw Data. Only file-to-file copies are supported.</description></item>
	/// <item><term>TLG</term><description>Linked Zones. Both files and DB are supported for the source as well as for the destination. In input, TLB = TLG, full information of base tracks only; 
    /// TLS = TLG, only geometrical information of base tracks.</description></item>
    /// <item><term>AI</term><description>Alignment Ignore indices for Linked Zones. Supported for destination only (text files).</description></item>
    /// <item><term>BI</term><description>Base Track indices for Linked Zones. Supported for destination only (text files).</description></item>
    /// <item><term>DBMI</term><description>DB MicroTrack indices for Linked Zones. Supported for destination only (text files).</description></item>
	/// <item><term>TSR</term><description>TotalScan Reconstructions. Both files and DB are supported for the source as well as for the destination.</description></item>
	/// <item><term>XML</term><description>ProgramSettings. Both files and DB are supported for the source as well as for the destination.</description></item>
	/// </list>	
	/// </para>
	/// <para>
	/// Usage examples:
	/// <list type="bullet">
	/// <item><term><b>Extracting a zone from the DB: </b><c>OperaCopy.exe db:\8\1024535288.tlg c:\data\myfile.tlg</c></term></item>
	/// <item><term><b>Extracting a TotalScan Reconstruction from the DB: </b><c>OperaCopy.exe db:\8\8032554443.tsr c:\data\myfile.tsr</c></term></item>
	/// <item><term><b>Writing a TotalScan Reconstruction to the DB: </b><c>OperaCopy.exe c:\data\myfile.tsr db:\24\100048833\1.tsr</c></term></item>
	/// </list>
	/// </para>
	/// </remarks>
	public class Exe
	{
		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main(string[] args)
		{
			//
			// TODO: Add code to start application here
			//
#if (!(DEBUG))
			try
			{
#endif
				if (args.Length != 2)
				{
					Console.WriteLine("usage: OperaCopy <source> <dest>");
					Console.WriteLine("supported types: TLG, TSR, RWC, RWD, XML");

					Console.WriteLine("TLG -------- input");
					Console.WriteLine(SySal.OperaPersistence.Help(typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone), true, true));
                    Console.WriteLine("TLB = TLG, full base track info; TLS = TLG, geometrical base track info only.");
					Console.WriteLine("TLG -------- output");
					Console.WriteLine(SySal.OperaPersistence.Help(typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone), false, true));
                    Console.WriteLine("AI -------- output");
                    Console.WriteLine("Alignment Ignore sections of TLG files translated to text files");
                    Console.WriteLine("BI -------- output");
                    Console.WriteLine("BaseTrack Index sections of TLG files translated to text files");
                    Console.WriteLine("DBMI -------- output");
                    Console.WriteLine("DB MicroTrack Index sections of TLG files translated to text files");

					Console.WriteLine("TSR -------- input");
					Console.WriteLine(SySal.OperaPersistence.Help(typeof(SySal.TotalScan.Volume), true, true));
					Console.WriteLine("TSR -------- output");
					Console.WriteLine(SySal.OperaPersistence.Help(typeof(SySal.TotalScan.Volume), false, true));

					Console.WriteLine("RWC -------- input");
					Console.WriteLine(SySal.OperaPersistence.Help(typeof(SySal.Scanning.Plate.IO.OPERA.RawData.Catalog), true, true));
					Console.WriteLine("RWC -------- output");
					Console.WriteLine(SySal.OperaPersistence.Help(typeof(SySal.Scanning.Plate.IO.OPERA.RawData.Catalog), false, true));

					Console.WriteLine("RWD -------- input");
					Console.WriteLine(SySal.OperaPersistence.Help(typeof(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment), true, true));
					Console.WriteLine("RWD -------- output");
					Console.WriteLine(SySal.OperaPersistence.Help(typeof(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment), false, true));

					Console.WriteLine("XML -------- input");
					Console.WriteLine(SySal.OperaPersistence.Help(typeof(SySal.OperaDb.ComputingInfrastructure.ProgramSettings), true, true));
					Console.WriteLine("XML -------- output");
					Console.WriteLine(SySal.OperaPersistence.Help(typeof(SySal.OperaDb.ComputingInfrastructure.ProgramSettings), false, true));

					return;
				}
				PersistenceType T = GetType(args[0]);
                PersistenceType Q = GetType(args[1]);
				if (T != Q && !(Q == PersistenceType.TLG && (T == PersistenceType.TLB || T == PersistenceType.TLS))) throw new Exception("Input and output persistence supports must be of the same type.");
				switch (T)
				{
                    case PersistenceType.TLG:
                        if (args[1].ToLower().EndsWith(".bi"))
                        {
                            SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIndex bi = (SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIndex)SySal.OperaPersistence.Restore(args[0], typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIndex));
                            System.IO.StreamWriter wr = new System.IO.StreamWriter(args[1]);
                            wr.Write("Index");
                            foreach (int ix in bi.Ids)
                            {
                                wr.WriteLine();
                                wr.Write(ix);
                            }
                            wr.Flush();
                            wr.Close();
                        }
                        else if (args[1].ToLower().EndsWith(".ai"))
                        {
                            SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIgnoreAlignment ai = (SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIgnoreAlignment)SySal.OperaPersistence.Restore(args[0], typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone.BaseTrackIgnoreAlignment));
                            System.IO.StreamWriter wr = new System.IO.StreamWriter(args[1]);
                            wr.Write("Index");
                            foreach (int ix in ai.Ids)
                            {
                                wr.WriteLine();
                                wr.Write(ix);
                            }
                            wr.Flush();
                            wr.Close();
                        }
                        else if (args[1].ToLower().EndsWith(".dbmi"))
                        {
                            SySal.OperaDb.Scanning.DBMIPMicroTrackIndex dbmi = (SySal.OperaDb.Scanning.DBMIPMicroTrackIndex)SySal.OperaPersistence.Restore(args[0], typeof(SySal.OperaDb.Scanning.DBMIPMicroTrackIndex));
                            System.IO.StreamWriter wr = new System.IO.StreamWriter(args[1]);
                            wr.Write("IdZone\tSide\tId");
                            foreach (SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex tx in dbmi.TopTracksIndex)
                            {
                                wr.WriteLine();
                                wr.Write(tx.ZoneId + "\t" + tx.Side + "\t" + tx.Id);
                            }
                            foreach (SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex tx in dbmi.BottomTracksIndex)
                            {
                                wr.WriteLine();
                                wr.Write(tx.ZoneId + "\t" + tx.Side + "\t" + tx.Id);
                            }
                            wr.Flush();
                            wr.Close();
                        }
                        else
                        {
                            SySal.OperaPersistence.LinkedZoneDetailLevel = SySal.OperaDb.Scanning.LinkedZone.DetailLevel.Full;
                            Console.WriteLine(SySal.OperaPersistence.Persist(args[1],
                            (SySal.Scanning.Plate.IO.OPERA.LinkedZone)SySal.OperaPersistence.Restore(args[0],
                            typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone))));
                        }
                        break;

                    case PersistenceType.TLB:
                        SySal.OperaPersistence.LinkedZoneDetailLevel = SySal.OperaDb.Scanning.LinkedZone.DetailLevel.BaseFull;
                        Console.WriteLine(SySal.OperaPersistence.Persist(args[1],
                        (SySal.Scanning.Plate.IO.OPERA.LinkedZone)SySal.OperaPersistence.Restore((args[0].Substring(0, args[0].Length - 1) + 'G'),
                        typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone))));
                        break;

                    case PersistenceType.TLS:
                        SySal.OperaPersistence.LinkedZoneDetailLevel = SySal.OperaDb.Scanning.LinkedZone.DetailLevel.BaseGeom;
                        Console.WriteLine(SySal.OperaPersistence.Persist(args[1],
                        (SySal.Scanning.Plate.IO.OPERA.LinkedZone)SySal.OperaPersistence.Restore((args[0].Substring(0, args[0].Length - 1) + 'G'),
                        typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone))));
                        break;
 
					case PersistenceType.TSR:	Console.WriteLine(SySal.OperaPersistence.Persist(args[1], 
													(SySal.TotalScan.Volume)SySal.OperaPersistence.Restore(args[0], 
													typeof(SySal.TotalScan.Volume)))); 
						break;

					case PersistenceType.RWC:	Console.WriteLine(SySal.OperaPersistence.Persist(args[1], 
													(SySal.Scanning.Plate.IO.OPERA.RawData.Catalog)SySal.OperaPersistence.Restore(args[0], 
													typeof(SySal.Scanning.Plate.IO.OPERA.RawData.Catalog)))); 
						break;

					case PersistenceType.RWD:	Console.WriteLine(SySal.OperaPersistence.Persist(args[1], 
													(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment)SySal.OperaPersistence.Restore(args[0], 
													typeof(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment)))); 
						break;

					case PersistenceType.XML:	Console.WriteLine(SySal.OperaPersistence.Persist(args[1], 
													(SySal.OperaDb.ComputingInfrastructure.ProgramSettings)SySal.OperaPersistence.Restore(args[0], 
													typeof(SySal.OperaDb.ComputingInfrastructure.ProgramSettings)))); 
						break;
				}
#if (!(DEBUG))
			}
			catch(Exception x)
			{
				Console.Error.WriteLine(x);
			}
#endif
		}

		private enum PersistenceType { TLG, TLB, TLS, TSR, RWC, RWD, XML }

		private static PersistenceType GetType(string path)
		{
			string p = path.ToLower();
            if (p.EndsWith(".ai")) return PersistenceType.TLG;
            if (p.EndsWith(".bi")) return PersistenceType.TLG;
            if (p.EndsWith(".dbmi")) return PersistenceType.TLG;
			if (p.EndsWith(".tlg")) return PersistenceType.TLG;
            if (p.EndsWith(".tlb")) return PersistenceType.TLB;
            if (p.EndsWith(".tls")) return PersistenceType.TLS;
			if (p.EndsWith(".tsr")) return PersistenceType.TSR;
			if (p.EndsWith(".rwc")) return PersistenceType.RWC;
			if (p.EndsWith(".rwd")) return PersistenceType.RWD;
			if (p.StartsWith(@"db:\xml\") || p.EndsWith(".xml")) return PersistenceType.XML;
			throw new Exception("Unknown type");
		}
	}
}
