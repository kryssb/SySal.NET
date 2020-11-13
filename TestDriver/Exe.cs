using System;
using SySal;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;
using SySal.DAQSystem.Drivers;
using System.Xml;
using System.Xml.Serialization;

namespace SySal.DAQSystem.Drivers.TestDriver
{
	/// <summary>
	/// Configuration for TestDriver.
	/// </summary>
	[Serializable]
	public class Config
	{
		/// <summary>
		/// The number of seconds to wait for each step.
		/// </summary>
		public int SecondsToWait;
	}

	/// <summary>
	/// TestDriver executor.
	/// </summary>
	/// <remarks>
	/// <para>
	/// TestDriver helps testing that BatchManager works correctly.
	/// </para>
	/// <para>
	/// It is intended as a diagnostic and debugging tool.
	/// </para>
	/// <para>
	/// Sample configuration for TestDriver:
	/// <example>
	/// <code>
	/// &lt;Config&gt;
	///  &lt;SecondsToWait&gt;3&lt;/SecondsToWait&gt;
	/// &lt;/Config&gt;
	/// </code>
	/// </example>
	/// </para>
	/// </remarks>
	public class Exe : MarshalByRefObject
	{
		/// <summary>
		/// Initializes the Lifetime Service.
		/// </summary>
		/// <returns>the lifetime service object or null.</returns>
		public override object InitializeLifetimeService()
		{
			return null;	
		}

		static SySal.DAQSystem.Drivers.HostEnv HE = null;
		
		static void ShowExplanation()
		{
			ExplanationForm EF = new ExplanationForm();
			System.IO.StringWriter strw = new System.IO.StringWriter();
			strw.WriteLine("TestDriver");
			strw.WriteLine("--------------");
			strw.WriteLine("TestDriver helps testing that BatchManager works correctly.");
			strw.WriteLine("It is intended as a diagnostic and debugging tool.");
			strw.WriteLine("--------------");
			strw.WriteLine("Sample configuration:");
			TestDriver.Config C = new TestDriver.Config();
			C.SecondsToWait = 10;
			System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(TestDriver.Config));			
			xmls.Serialize(strw, C);
			EF.RTFOut.Text = strw.ToString();
			EF.ShowDialog();			
		}

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[MTAThread]
		static void Main(string[] args)
		{
			HE = SySal.DAQSystem.Drivers.HostEnv.Own;
			if (HE == null)
			{
				ShowExplanation();
				return;
			}
			try
			{
				SySal.DAQSystem.Drivers.VolumeOperationInfo VInfo = (SySal.DAQSystem.Drivers.VolumeOperationInfo)HE.StartupInfo;
				SySal.OperaDb.OperaDbConnection DB = new SySal.OperaDb.OperaDbConnection(VInfo.DBServers, VInfo.DBUserName, VInfo.DBPassword);
				DB.Open();
				long newdrv = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT ID FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE = 'TestDriver.exe' AND DRIVERLEVEL < 2 ORDER BY DRIVERLEVEL DESC", DB).ExecuteScalar());
				SySal.DAQSystem.Drivers.ScanningStartupInfo SInfo = new SySal.DAQSystem.Drivers.ScanningStartupInfo();
				SInfo.MachineId = VInfo.MachineId;
				SInfo.Notes = VInfo.Notes;
				SInfo.ProgramSettingsId = newdrv;
				SInfo.Plate = new SySal.DAQSystem.Scanning.MountPlateDesc();
				SInfo.Plate.BrickId = VInfo.BrickId;
				SInfo.Plate.PlateId = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT MIN(ID) FROM VW_PLATES WHERE ID_EVENTBRICK = " + VInfo.BrickId, DB).ExecuteScalar());
				DB.Close();
				long newop = HE.Start(SInfo);
				HE.WriteLine("Now waiting!");
				HE.Wait(newop);
				HE.WriteLine("OK!");
				HE.Complete = true;
			}
			catch (Exception x)
			{
				System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(TestDriver.Config));
				Config C = (Config)xmls.Deserialize(new System.IO.StringReader(HE.ProgramSettings));
				HE.Progress = 0.0;
				HE.WriteLine("Waiting " + C.SecondsToWait + " seconds before next message...");
				System.Threading.Thread.Sleep(C.SecondsToWait * 1000);
				HE.Progress = 0.5;
				HE.WriteLine("Now waiting for " + C.SecondsToWait + " seconds before exiting...");
				System.Threading.Thread.Sleep(C.SecondsToWait * 1000);
				HE.Progress = 1.0;
			}
			HE.WriteLine("Exit!");
			HE.Complete = true;
		}
	}
}
