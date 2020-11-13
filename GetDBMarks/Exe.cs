using System;

namespace SySal.Executables.GetDBMarks
{
	/// <summary>
	/// GetDBMarks - Command line tool to read mark sets from the DB in a format that can be used for scanning.
	/// </summary>
	/// <remarks>
	/// <para>Usage: <c>GetDBMarks.exe &lt;brick id&gt; &lt;plate id&gt; &lt;type-or-id&gt;</c></para>
	/// <para><c>type</c> can be either <c>nominal</c> or <c>calibrated</c>. The explicit Id of the calibration can be given as an alternative.</para>
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
			if (args.Length != 3 && args.Length != 4)
			{
				Console.WriteLine("usage: GetDBMarks <brick id> <plate id> <type-or-id> <mark set>"); 
				Console.WriteLine("<type> can be either nominal or calibrated; as an alternative, the explicit Id of the calibration requested can be given.");
                Console.WriteLine("<mark set> can be " + SySal.DAQSystem.Drivers.MarkChar.SpotOptical + " (Spot Optical), " + SySal.DAQSystem.Drivers.MarkChar.LineXRay + " (X-ray lateral), " + SySal.DAQSystem.Drivers.MarkChar.SpotXRay + " (X-ray for CS-brick connection). This parameter is needed when the id is not directly specified.");
				return;
			}
			long id = 0;
            char mt = SySal.DAQSystem.Drivers.MarkChar.None;
			bool calibrated = String.Compare(args[2], "calibrated", true) == 0;
            if (!calibrated && String.Compare(args[2], "nominal", true) != 0)
                id = Convert.ToInt64(args[2]);
            else
            {
                mt = Convert.ToChar(args[3]);
            }            
			SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
			SySal.OperaDb.OperaDbConnection conn = new SySal.OperaDb.OperaDbConnection(cred.DBServer, cred.DBUserName, cred.DBPassword);
			conn.Open();
			if (id == 0)
			{
				long calibrationid;
				Console.WriteLine(SySal.OperaDb.Scanning.Utilities.GetMapString(Convert.ToInt64(args[0]), Convert.ToInt64(args[1]), calibrated, SySal.OperaDb.Scanning.Utilities.CharToMarkType(mt), out calibrationid, conn, null));
			}
			else
				Console.WriteLine(SySal.OperaDb.Scanning.Utilities.GetMapString(Convert.ToInt64(args[0]), Convert.ToInt64(args[1]), id, conn, null));
			conn.Close();
		}
	}
}
