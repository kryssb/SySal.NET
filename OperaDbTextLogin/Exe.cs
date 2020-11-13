using System;

namespace SySal.Executables.OperaDbTextLogin
{
	/// <summary>
	/// OperaDbTextLogin - Command line tool to set the default credential record of the current user.
	/// </summary>
	/// <remarks>
	/// <para>Every user on a workstation/server can have his/her own default credential record for the OPERA DB and Computing Infrastructure. 
	/// This records saves continuous login requests on the OPERA DB and on the Computing Infrastructure services.</para>
	/// <para>The record is saved in the user profile in encrypted form.</para>
	/// <para>OperaDbTextLogin creates or overwrites the default credential record.</para>
	/// <para>The command line should have the form: <c>OperaDbTextLogin.exe operadbtextlogin &lt;dbserver(s)&gt; &lt;dbusername&gt; &lt;dbpassword&gt; &lt;operadbusername&gt; &lt;operadbpassword&gt;</c></para>
	/// <para>More than one DB server can be specified. The server name is its TNS name. Multiple server names must be separated by commas (',').</para>
	/// <para>Strings must be enclosed within quotes ("") if they contain spaces</para>
	/// <para><b>CAUTION: passwords appear in plain text, since they are typed on the OS shell command line. Ensure that nobody is looking at the screen, and be aware that the command list might be browsed.</b></para>
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
			if (args.Length != 5)
			{
				Console.WriteLine("usage: operadbtextlogin <dbserver(s)> <dbusername> <dbpassword> <operadbusername> <operadbpassword>");
				Console.WriteLine("Strings must be enclosed within quotes (\"\") if they contain spaces");
				return;
			}

			SySal.OperaDb.OperaDbCredentials o = new SySal.OperaDb.OperaDbCredentials();
			o.DBServer = args[0];
			o.DBUserName = args[1];
			o.DBPassword = args[2];
			o.OPERAUserName = args[3];
			o.OPERAPassword = args[4];
			o.Record();
		}
	}
}
