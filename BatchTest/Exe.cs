using System;

namespace SySal.Executables.BatchTest
{
	/// <summary>
	/// Main ex.
	/// </summary>
	public class Exe
	{
		/// <summary>
		/// Batch Test - test program for data processing servers.
		/// </summary>
		/// <remarks>
		/// <para>This is a harmless executable used to test DataProcessingServers.</para>
		/// <para>Keeps the CPU busy for the time specified (in seconds).</para>
		/// <para>Usage: <c>BatchTest.exe &lt;time to wait in seconds&gt;</c></para>
		/// </remarks>
		[STAThread]
		static void Main(string[] args)
		{
			//
			// TODO: Add code to start application here
			//
			if (args.Length != 1 && args.Length != 6)
			{
				Console.WriteLine("Batch Test - test program for data processing servers.");
				Console.WriteLine("Keeps the CPU busy for the time specified (in seconds).");
				Console.WriteLine("usage: batchtest <time to wait in seconds>");
				Console.WriteLine("or: batchtest <time to wait in seconds> <dbserver> <dbuser> <dbpwd> <operauser> <operapwd>");
				return;
			}
            if (args.Length == 6)
            {
                SySal.OperaDb.OperaDbCredentials cred = new SySal.OperaDb.OperaDbCredentials();
                cred.DBServer = args[1];
                cred.DBUserName = args[2];
                cred.DBPassword = args[3];
                cred.OPERAUserName = args[4];
                cred.OPERAPassword = args[5];
                System.Diagnostics.Process proc = new System.Diagnostics.Process();
                proc.StartInfo.UseShellExecute = false;
                cred.RecordToEnvironment(proc.StartInfo.EnvironmentVariables);
                proc.StartInfo.FileName = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName;
                proc.StartInfo.Arguments = args[0];                
                Console.WriteLine("Spawning child and closing");
                proc.Start();
                proc.WaitForExit();
                return;
            }
            string dbserver = "";
            string dbuser = "";
            string operauser = "";
            try
            {
                SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
                dbserver = cred.DBServer;
                dbuser = cred.DBUserName;
                operauser = cred.OPERAUserName;
                SySal.OperaDb.OperaDbConnection conn = cred.Connect();
                conn.Open();                
                Console.WriteLine("DBTest: " + new SySal.OperaDb.OperaDbCommand("SELECT * FROM DUAL", conn).ExecuteScalar().ToString());
                Console.Write("Check access: ");
                Console.WriteLine(SySal.OperaDb.ComputingInfrastructure.User.CheckLogin(operauser, cred.OPERAPassword, conn, null));
            }
            catch (Exception x) 
            {
                Console.WriteLine("Exception: " + x.ToString());
            }

            Console.WriteLine("Running with:\r\nDBServer-> " + dbserver + "\r\nDBUser-> " + dbuser + "\r\nOPERAUser-> " + operauser);
            Console.WriteLine("TEMPDIR=\"" + System.Environment.GetEnvironmentVariable("TEMP") + "\"");
            string escalatefile = System.Environment.ExpandEnvironmentVariables("%TEMP%/batchtest.escalate");
            Console.WriteLine("Create the file: \"" + escalatefile + "\" to escalate the powerclass.");
			System.DateTime end = System.DateTime.Now.AddSeconds(Convert.ToDouble(args[0]));
			while (System.DateTime.Now.CompareTo(end) < 0);
            if (System.IO.File.Exists(escalatefile)) throw new System.OutOfMemoryException("Batch escalation required.");
		}
	}
}
