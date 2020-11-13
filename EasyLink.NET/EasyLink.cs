using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;
using System.IO;
using SySal;
using SySal.Management;
using SySal.Scanning;
using SySal.Scanning.Plate.IO.OPERA.RawData;
using SySal.Scanning.PostProcessing;
using SySal.Scanning.PostProcessing.FragmentLinking;
using System.Reflection;
using System.Threading;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Messaging;

namespace SySal.Executables.EasyLinkNET
{
	/// <summary>
	/// GUI tool for interactive sessions of raw data linking to TLG files.
	/// </summary>
	/// <remarks>This executable is almost obsolete. Use of BatchLink (See <see cref="SySal.Executables.BatchLink.Exe"/>) is recommended instead.</remarks>
	public class Exe : System.Windows.Forms.Form
	{
		private SySal.Scanning.Plate.LinkedZone Ret;

		public void LinkResult(IAsyncResult ar)
		{
			AsyncResult aRes = (AsyncResult)ar;
			LinkCallback el = (LinkCallback)aRes.AsyncDelegate;
			el.EndInvoke(ar);
			ControlEnabler(false);
		}

		void ControlEnabler(bool running)
		{
			CatalogFile.Enabled = !running;
			OutputFile.Enabled = !running;
			ConfigFile.Enabled = !running;
			AssemblyFile.Enabled = !running;
			SelCatalogFile.Enabled = !running;
			SelOutputFile.Enabled = !running;
			SelConfigFile.Enabled = !running;
			SelAssemblyFile.Enabled = !running;
			LoadConfig.Enabled = !running;
			SaveConfig.Enabled = !running;
			EditConfig.Enabled = !running;
			LoadAssembly.Enabled = !running;
			Start.Enabled = !running;
			TopSlopeMultiplier.Enabled = !running;
			BottomSlopeMultiplier.Enabled = !running;
			Stop.Enabled = running;
			TopSlopeDX.Enabled = !running;
			TopSlopeDY.Enabled = !running;
			BottomSlopeDX.Enabled = !running;
			BottomSlopeDY.Enabled = !running;
		}

		public struct EasyLinkConfig
		{
			public string CatalogFile;
			public string OutputFile;
			public string ConfigFile;
			public string AssemblyFile;
			public double TopSlopeMultiplier;
			public double BottomSlopeMultiplier;
			public double TopSlopeDX, TopSlopeDY;
			public double BottomSlopeDX, BottomSlopeDY;
		}
	
		private EasyLinkConfig MyConfig;

		private string mycfgfile;

		private System.Windows.Forms.TextBox CatalogFile;
		private System.Windows.Forms.TextBox OutputFile;
		private System.Windows.Forms.TextBox ConfigFile;
		private System.Windows.Forms.TextBox AssemblyFile;
		private System.Windows.Forms.Button LoadConfig;
		private System.Windows.Forms.Button SaveConfig;
		private System.Windows.Forms.Button LoadAssembly;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		private System.Windows.Forms.ProgressBar RecProgress;
		private System.Windows.Forms.Button EditConfig;

		private SySal.Scanning.PostProcessing.FragmentLinking.IFragmentLinker ifl = null;
		private System.Windows.Forms.Button SelConfigFile;
		private System.Windows.Forms.Button SelAssemblyFile;
		private System.Windows.Forms.Button SelOutputFile;
		private System.Windows.Forms.Button SelCatalogFile;
		private System.Windows.Forms.Button Start;
		private System.Windows.Forms.Button Stop;
		private SySal.Management.IManageable imng = null;

		private Thread RecThread = null;
		private System.Windows.Forms.RadioButton CHORUSFormat;
		private System.Windows.Forms.RadioButton OPERAFormat;
		private System.Windows.Forms.Label label1;
		private bool StopThread = false;
		private System.Windows.Forms.TextBox TopSlopeMultiplier;
		private System.Windows.Forms.TextBox BottomSlopeMultiplier;
		private System.Windows.Forms.Button cmdComputeShrink;
		private System.Windows.Forms.TextBox TopSlopeDX;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.TextBox TopSlopeDY;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.TextBox BottomSlopeDY;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.TextBox BottomSlopeDX;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.Label label2;

		public Exe()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			RecProgress.Minimum = 0;
			RecProgress.Maximum = 100;
			RecProgress.Value = 0;
			mycfgfile = System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName;
			mycfgfile = mycfgfile.Remove(mycfgfile.Length - 3, 3) + "cfg";
			MyConfig.CatalogFile = "";
			MyConfig.OutputFile = "";
			MyConfig.AssemblyFile = "";
			MyConfig.ConfigFile = "";
			MyConfig.TopSlopeMultiplier = 1.0;
			MyConfig.BottomSlopeMultiplier = 1.0;
			MyConfig.TopSlopeDX = 0.0;
			MyConfig.TopSlopeDY = 0.0;
			MyConfig.BottomSlopeDX = 0.0;
			MyConfig.BottomSlopeDY = 0.0;
			CHORUSFormat.Checked = false;
			OPERAFormat.Checked = true;
		}

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		protected override void Dispose( bool disposing )
		{
			if( disposing )
			{
				if (components != null) 
				{
					components.Dispose();
				}
			}
			base.Dispose( disposing );
		}

		#region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.CatalogFile = new System.Windows.Forms.TextBox();
			this.OutputFile = new System.Windows.Forms.TextBox();
			this.ConfigFile = new System.Windows.Forms.TextBox();
			this.AssemblyFile = new System.Windows.Forms.TextBox();
			this.LoadConfig = new System.Windows.Forms.Button();
			this.SaveConfig = new System.Windows.Forms.Button();
			this.LoadAssembly = new System.Windows.Forms.Button();
			this.RecProgress = new System.Windows.Forms.ProgressBar();
			this.EditConfig = new System.Windows.Forms.Button();
			this.SelConfigFile = new System.Windows.Forms.Button();
			this.SelAssemblyFile = new System.Windows.Forms.Button();
			this.SelOutputFile = new System.Windows.Forms.Button();
			this.SelCatalogFile = new System.Windows.Forms.Button();
			this.Start = new System.Windows.Forms.Button();
			this.Stop = new System.Windows.Forms.Button();
			this.CHORUSFormat = new System.Windows.Forms.RadioButton();
			this.OPERAFormat = new System.Windows.Forms.RadioButton();
			this.label1 = new System.Windows.Forms.Label();
			this.TopSlopeMultiplier = new System.Windows.Forms.TextBox();
			this.BottomSlopeMultiplier = new System.Windows.Forms.TextBox();
			this.label2 = new System.Windows.Forms.Label();
			this.cmdComputeShrink = new System.Windows.Forms.Button();
			this.TopSlopeDX = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.TopSlopeDY = new System.Windows.Forms.TextBox();
			this.label4 = new System.Windows.Forms.Label();
			this.BottomSlopeDY = new System.Windows.Forms.TextBox();
			this.label6 = new System.Windows.Forms.Label();
			this.BottomSlopeDX = new System.Windows.Forms.TextBox();
			this.label7 = new System.Windows.Forms.Label();
			this.SuspendLayout();
			// 
			// CatalogFile
			// 
			this.CatalogFile.Location = new System.Drawing.Point(144, 8);
			this.CatalogFile.Name = "CatalogFile";
			this.CatalogFile.Size = new System.Drawing.Size(312, 20);
			this.CatalogFile.TabIndex = 1;
			this.CatalogFile.Text = "";
			// 
			// OutputFile
			// 
			this.OutputFile.Location = new System.Drawing.Point(144, 40);
			this.OutputFile.Name = "OutputFile";
			this.OutputFile.Size = new System.Drawing.Size(312, 20);
			this.OutputFile.TabIndex = 4;
			this.OutputFile.Text = "";
			// 
			// ConfigFile
			// 
			this.ConfigFile.Location = new System.Drawing.Point(144, 72);
			this.ConfigFile.Name = "ConfigFile";
			this.ConfigFile.Size = new System.Drawing.Size(312, 20);
			this.ConfigFile.TabIndex = 6;
			this.ConfigFile.Text = "";
			// 
			// AssemblyFile
			// 
			this.AssemblyFile.Location = new System.Drawing.Point(144, 104);
			this.AssemblyFile.Name = "AssemblyFile";
			this.AssemblyFile.Size = new System.Drawing.Size(312, 20);
			this.AssemblyFile.TabIndex = 8;
			this.AssemblyFile.Text = "";
			// 
			// LoadConfig
			// 
			this.LoadConfig.Location = new System.Drawing.Point(16, 200);
			this.LoadConfig.Name = "LoadConfig";
			this.LoadConfig.Size = new System.Drawing.Size(112, 24);
			this.LoadConfig.TabIndex = 9;
			this.LoadConfig.Text = "Load Config";
			this.LoadConfig.Click += new System.EventHandler(this.LoadConfig_Click);
			// 
			// SaveConfig
			// 
			this.SaveConfig.Location = new System.Drawing.Point(16, 232);
			this.SaveConfig.Name = "SaveConfig";
			this.SaveConfig.Size = new System.Drawing.Size(112, 24);
			this.SaveConfig.TabIndex = 10;
			this.SaveConfig.Text = "Save Config";
			this.SaveConfig.Click += new System.EventHandler(this.SaveConfig_Click);
			// 
			// LoadAssembly
			// 
			this.LoadAssembly.Location = new System.Drawing.Point(16, 168);
			this.LoadAssembly.Name = "LoadAssembly";
			this.LoadAssembly.Size = new System.Drawing.Size(112, 24);
			this.LoadAssembly.TabIndex = 11;
			this.LoadAssembly.Text = "Load Assembly";
			this.LoadAssembly.Click += new System.EventHandler(this.LoadAssembly_Click);
			// 
			// RecProgress
			// 
			this.RecProgress.Location = new System.Drawing.Point(16, 136);
			this.RecProgress.Name = "RecProgress";
			this.RecProgress.Size = new System.Drawing.Size(440, 16);
			this.RecProgress.TabIndex = 12;
			// 
			// EditConfig
			// 
			this.EditConfig.Location = new System.Drawing.Point(16, 264);
			this.EditConfig.Name = "EditConfig";
			this.EditConfig.Size = new System.Drawing.Size(112, 24);
			this.EditConfig.TabIndex = 13;
			this.EditConfig.Text = "Edit Config";
			this.EditConfig.Click += new System.EventHandler(this.EditConfig_Click);
			// 
			// SelConfigFile
			// 
			this.SelConfigFile.Location = new System.Drawing.Point(16, 72);
			this.SelConfigFile.Name = "SelConfigFile";
			this.SelConfigFile.Size = new System.Drawing.Size(112, 24);
			this.SelConfigFile.TabIndex = 14;
			this.SelConfigFile.Text = "Config File";
			this.SelConfigFile.Click += new System.EventHandler(this.SelConfigFile_Click);
			// 
			// SelAssemblyFile
			// 
			this.SelAssemblyFile.Location = new System.Drawing.Point(16, 104);
			this.SelAssemblyFile.Name = "SelAssemblyFile";
			this.SelAssemblyFile.Size = new System.Drawing.Size(112, 24);
			this.SelAssemblyFile.TabIndex = 15;
			this.SelAssemblyFile.Text = "Assembly File";
			this.SelAssemblyFile.Click += new System.EventHandler(this.SelAssemblyFile_Click);
			// 
			// SelOutputFile
			// 
			this.SelOutputFile.Location = new System.Drawing.Point(16, 40);
			this.SelOutputFile.Name = "SelOutputFile";
			this.SelOutputFile.Size = new System.Drawing.Size(112, 24);
			this.SelOutputFile.TabIndex = 16;
			this.SelOutputFile.Text = "Output File";
			this.SelOutputFile.Click += new System.EventHandler(this.SelOutputFile_Click);
			// 
			// SelCatalogFile
			// 
			this.SelCatalogFile.Location = new System.Drawing.Point(16, 8);
			this.SelCatalogFile.Name = "SelCatalogFile";
			this.SelCatalogFile.Size = new System.Drawing.Size(112, 24);
			this.SelCatalogFile.TabIndex = 17;
			this.SelCatalogFile.Text = "Catalog File";
			this.SelCatalogFile.Click += new System.EventHandler(this.SelCatalogFile_Click);
			// 
			// Start
			// 
			this.Start.Location = new System.Drawing.Point(344, 168);
			this.Start.Name = "Start";
			this.Start.Size = new System.Drawing.Size(112, 24);
			this.Start.TabIndex = 18;
			this.Start.Text = "Start";
			this.Start.Click += new System.EventHandler(this.Start_Click);
			// 
			// Stop
			// 
			this.Stop.Location = new System.Drawing.Point(344, 200);
			this.Stop.Name = "Stop";
			this.Stop.Size = new System.Drawing.Size(112, 24);
			this.Stop.TabIndex = 19;
			this.Stop.Text = "Stop";
			this.Stop.Click += new System.EventHandler(this.Stop_Click);
			// 
			// CHORUSFormat
			// 
			this.CHORUSFormat.Location = new System.Drawing.Point(144, 168);
			this.CHORUSFormat.Name = "CHORUSFormat";
			this.CHORUSFormat.Size = new System.Drawing.Size(152, 24);
			this.CHORUSFormat.TabIndex = 20;
			this.CHORUSFormat.Text = "CHORUS file format";
			// 
			// OPERAFormat
			// 
			this.OPERAFormat.Location = new System.Drawing.Point(144, 200);
			this.OPERAFormat.Name = "OPERAFormat";
			this.OPERAFormat.Size = new System.Drawing.Size(152, 24);
			this.OPERAFormat.TabIndex = 21;
			this.OPERAFormat.Text = "OPERA file format";
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(144, 232);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(104, 24);
			this.label1.TabIndex = 22;
			this.label1.Text = "Top slope mult.";
			this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// TopSlopeMultiplier
			// 
			this.TopSlopeMultiplier.Location = new System.Drawing.Point(264, 232);
			this.TopSlopeMultiplier.Name = "TopSlopeMultiplier";
			this.TopSlopeMultiplier.Size = new System.Drawing.Size(40, 20);
			this.TopSlopeMultiplier.TabIndex = 23;
			this.TopSlopeMultiplier.Text = "1";
			this.TopSlopeMultiplier.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// BottomSlopeMultiplier
			// 
			this.BottomSlopeMultiplier.Location = new System.Drawing.Point(264, 264);
			this.BottomSlopeMultiplier.Name = "BottomSlopeMultiplier";
			this.BottomSlopeMultiplier.Size = new System.Drawing.Size(40, 20);
			this.BottomSlopeMultiplier.TabIndex = 25;
			this.BottomSlopeMultiplier.Text = "1";
			this.BottomSlopeMultiplier.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(144, 264);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(104, 24);
			this.label2.TabIndex = 24;
			this.label2.Text = "Bottom slope mult.";
			this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// cmdComputeShrink
			// 
			this.cmdComputeShrink.Location = new System.Drawing.Point(344, 232);
			this.cmdComputeShrink.Name = "cmdComputeShrink";
			this.cmdComputeShrink.Size = new System.Drawing.Size(112, 24);
			this.cmdComputeShrink.TabIndex = 26;
			this.cmdComputeShrink.Text = "Compute Shrinkage";
			this.cmdComputeShrink.Click += new System.EventHandler(this.cmdComputeShrink_Click);
			// 
			// TopSlopeDX
			// 
			this.TopSlopeDX.Location = new System.Drawing.Point(264, 296);
			this.TopSlopeDX.Name = "TopSlopeDX";
			this.TopSlopeDX.Size = new System.Drawing.Size(40, 20);
			this.TopSlopeDX.TabIndex = 28;
			this.TopSlopeDX.Text = "0";
			this.TopSlopeDX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label3
			// 
			this.label3.Location = new System.Drawing.Point(144, 296);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(112, 24);
			this.label3.TabIndex = 27;
			this.label3.Text = "Top slope delta (X)";
			this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// TopSlopeDY
			// 
			this.TopSlopeDY.Location = new System.Drawing.Point(416, 296);
			this.TopSlopeDY.Name = "TopSlopeDY";
			this.TopSlopeDY.Size = new System.Drawing.Size(40, 20);
			this.TopSlopeDY.TabIndex = 30;
			this.TopSlopeDY.Text = "0";
			this.TopSlopeDY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label4
			// 
			this.label4.Location = new System.Drawing.Point(304, 296);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(104, 24);
			this.label4.TabIndex = 29;
			this.label4.Text = "Top slope delta (Y)";
			this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// BottomSlopeDY
			// 
			this.BottomSlopeDY.Location = new System.Drawing.Point(416, 320);
			this.BottomSlopeDY.Name = "BottomSlopeDY";
			this.BottomSlopeDY.Size = new System.Drawing.Size(40, 20);
			this.BottomSlopeDY.TabIndex = 34;
			this.BottomSlopeDY.Text = "0";
			this.BottomSlopeDY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label6
			// 
			this.label6.Location = new System.Drawing.Point(304, 320);
			this.label6.Name = "label6";
			this.label6.Size = new System.Drawing.Size(120, 24);
			this.label6.TabIndex = 33;
			this.label6.Text = "Bottom slope delta (Y)";
			this.label6.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// BottomSlopeDX
			// 
			this.BottomSlopeDX.Location = new System.Drawing.Point(264, 320);
			this.BottomSlopeDX.Name = "BottomSlopeDX";
			this.BottomSlopeDX.Size = new System.Drawing.Size(40, 20);
			this.BottomSlopeDX.TabIndex = 32;
			this.BottomSlopeDX.Text = "0";
			this.BottomSlopeDX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label7
			// 
			this.label7.Location = new System.Drawing.Point(144, 320);
			this.label7.Name = "label7";
			this.label7.Size = new System.Drawing.Size(120, 24);
			this.label7.TabIndex = 31;
			this.label7.Text = "Bottom slope delta (X)";
			this.label7.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// Exe
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(462, 348);
			this.Controls.AddRange(new System.Windows.Forms.Control[] {
																		  this.BottomSlopeDY,
																		  this.label6,
																		  this.BottomSlopeDX,
																		  this.label7,
																		  this.TopSlopeDY,
																		  this.label4,
																		  this.TopSlopeDX,
																		  this.label3,
																		  this.cmdComputeShrink,
																		  this.BottomSlopeMultiplier,
																		  this.label2,
																		  this.TopSlopeMultiplier,
																		  this.label1,
																		  this.OPERAFormat,
																		  this.CHORUSFormat,
																		  this.Stop,
																		  this.Start,
																		  this.SelCatalogFile,
																		  this.SelOutputFile,
																		  this.SelAssemblyFile,
																		  this.SelConfigFile,
																		  this.EditConfig,
																		  this.RecProgress,
																		  this.LoadAssembly,
																		  this.SaveConfig,
																		  this.LoadConfig,
																		  this.AssemblyFile,
																		  this.ConfigFile,
																		  this.OutputFile,
																		  this.CatalogFile});
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
			this.MaximizeBox = false;
			this.Name = "Exe";
			this.Text = "Exe.NET";
			this.Closing += new System.ComponentModel.CancelEventHandler(this.EasyLink_Closing);
			this.Load += new System.EventHandler(this.EasyLink_Load);
			this.ResumeLayout(false);

		}
		#endregion

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[MTAThread]
		static void Main() 
		{
			Application.Run(new Exe());
		}

		private void EasyLink_Load(object sender, System.EventArgs e)
		{
			System.IO.FileStream f = null;
			try
			{
				f = new System.IO.FileStream(mycfgfile, FileMode.Open);
				System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(MyConfig.GetType());
				MyConfig = (EasyLinkConfig)xmls.Deserialize(f);
				f.Close();
				CatalogFile.Text = MyConfig.CatalogFile;
				AssemblyFile.Text = MyConfig.AssemblyFile;
				ConfigFile.Text = MyConfig.ConfigFile;
				OutputFile.Text = MyConfig.OutputFile;
				TopSlopeMultiplier.Text = MyConfig.TopSlopeMultiplier.ToString();
				BottomSlopeMultiplier.Text = MyConfig.BottomSlopeMultiplier.ToString();
				TopSlopeDX.Text = MyConfig.TopSlopeDX.ToString();
				TopSlopeDY.Text = MyConfig.TopSlopeDY.ToString();
				BottomSlopeDX.Text = MyConfig.BottomSlopeDX.ToString();
				BottomSlopeDY.Text = MyConfig.BottomSlopeDY.ToString();
			}
			catch (System.Exception) {};
			if (f != null) f.Close();
		}

		private void EasyLink_Closing(object sender, System.ComponentModel.CancelEventArgs e)
		{
			System.IO.FileStream f = null;
			MyConfig.CatalogFile = CatalogFile.Text;
			MyConfig.AssemblyFile = AssemblyFile.Text;
			MyConfig.ConfigFile = ConfigFile.Text;
			MyConfig.OutputFile = OutputFile.Text;
			MyConfig.TopSlopeMultiplier = Convert.ToDouble(TopSlopeMultiplier.Text);
			MyConfig.BottomSlopeMultiplier = Convert.ToDouble(BottomSlopeMultiplier.Text);
			MyConfig.TopSlopeDX = Convert.ToDouble(TopSlopeDX.Text);
			MyConfig.TopSlopeDY = Convert.ToDouble(TopSlopeDY.Text);
			MyConfig.BottomSlopeDX = Convert.ToDouble(BottomSlopeDX.Text);
			MyConfig.BottomSlopeDY = Convert.ToDouble(BottomSlopeDY.Text);
			try
			{
				f = new System.IO.FileStream(mycfgfile, FileMode.Create, FileAccess.Write, FileShare.None);
				System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(MyConfig.GetType());
				xmls.Serialize(f, MyConfig);
				f.Close();
			}
			catch (System.Exception x) 
			{
				MessageBox.Show(x.ToString(), "Error");
			};
			if (f != null) f.Close();		
		}

		private void LoadAssemblyModule()
		{
			Assembly a;
			try
			{
				a = Assembly.LoadFrom(AssemblyFile.Text);
				System.Type [] types = a.GetExportedTypes();
				foreach	(System.Type t in types)
				{
					System.Type [] intfcs = t.GetInterfaces();
					foreach (System.Type u in intfcs)
						if (u.ToString() == "SySal.Scanning.PostProcessing.FragmentLinking.IFragmentLinker")
						{
							foreach (System.Type v in intfcs)
								if (v.ToString() == "SySal.Management.IManageable")
								{
									object o = a.CreateInstance(t.FullName);
									ifl = (IFragmentLinker)o;
									imng = (IManageable)o;
									ifl.Load = new dLoadFragment(LoadFragment);
									ifl.Progress = new dProgress(Progress);
									ifl.ShouldStop = new dShouldStop(ShouldStop);
									return;
								}
						};
				}
			}
			catch (System.Exception x) 
			{
				MessageBox.Show(x.ToString(), "Can't load assembly");
			};
		}

		public bool ShouldStop()
		{
			return StopThread;
		}

		public SySal.Scanning.Plate.IO.OPERA.RawData.Fragment LoadFragment(uint index)
		{
			System.IO.FileStream f;
			int i;
			string basetext = CatalogFile.Text.Substring(0, CatalogFile.Text.Length - 1) + "d.";
			f = new System.IO.FileStream(basetext + Convert.ToString(index, 16).PadLeft(8, '0'), FileMode.Open, FileAccess.Read);
			Fragment Frag = new Fragment(f);
			for (i = 0; i < Frag.Length; i++)
			{
				Fragment.View v = Frag[i];
				int j, l;
				l = v.Top.Length;
				for (j = 0; j < l; j++)
					MIPEmulsionTrack.AdjustSlopes(v.Top[j], MyConfig.TopSlopeMultiplier, MyConfig.TopSlopeDX, MyConfig.TopSlopeDY);
				l = v.Bottom.Length;
				for (j = 0; j < l; j++)
					MIPEmulsionTrack.AdjustSlopes(v.Bottom[j], MyConfig.BottomSlopeMultiplier, MyConfig.BottomSlopeDX, MyConfig.BottomSlopeDY);
			}
			f.Close();
			return Frag;
		}

		public void Progress(double percent)
		{
			RecProgress.Value = (int)(percent);	
		}

		private void EditConfig_Click(object sender, System.EventArgs e)
		{
			if (ifl == null || imng == null) LoadAssemblyModule();
			SySal.Management.Configuration tempc = imng.Config;
			if (imng.EditConfiguration(ref tempc)) imng.Config = tempc;
		}

		private void SaveConfig_Click(object sender, System.EventArgs e)
		{
			if (ifl == null || imng == null) LoadAssemblyModule();
			System.IO.FileStream f = null;
			try
			{
				f = new FileStream(ConfigFile.Text, FileMode.Create);
				System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(imng.Config.GetType());
				xmls.Serialize(f, imng.Config);
				f.Close();
			}
			catch (Exception x)
			{
				if (f != null) f.Close();
				MessageBox.Show(x.Message);
			}
		}

		private void LoadConfig_Click(object sender, System.EventArgs e)
		{
			if (ifl == null || imng == null) LoadAssemblyModule();
			System.IO.FileStream f = null;
			try
			{
				f = new FileStream(ConfigFile.Text, FileMode.Open, FileAccess.Read);
				System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(imng.Config.GetType());
				SySal.Management.Configuration cnf = imng.Config;
				cnf = (SySal.Management.Configuration)xmls.Deserialize(f);
				imng.Config = cnf;
				f.Close();
			}
			catch (Exception x)
			{
				if (f != null) f.Close();
				MessageBox.Show(x.Message);
			}		
		}

		private void SelCatalogFile_Click(object sender, System.EventArgs e)
		{
			OpenFileDialog d = new OpenFileDialog();
			d.Filter = "Catalog files (*.rwc)|*.rwc|All files (*.*)|*.*";
			d.DefaultExt = "rwc";
			if (d.ShowDialog() == DialogResult.OK) CatalogFile.Text = d.FileName;
		}

		private void SelOutputFile_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog d = new SaveFileDialog();
			d.Filter = "Output files (*.tlg)|*.tlg|All files (*.*)|*.*";
			d.DefaultExt = "tlg";
			if (d.ShowDialog() == DialogResult.OK) OutputFile.Text = d.FileName;				
		}

		private void SelConfigFile_Click(object sender, System.EventArgs e)
		{
			OpenFileDialog d = new OpenFileDialog();
			d.Filter = "Config files (*.xml)|*.xml|All files (*.*)|*.*";
			d.DefaultExt = "xml";
			if (d.ShowDialog() == DialogResult.OK) ConfigFile.Text = d.FileName;		
		}

		private void SelAssemblyFile_Click(object sender, System.EventArgs e)
		{
			OpenFileDialog d = new OpenFileDialog();
			d.Filter = "Assembly files (*.dll)|*.dll|All files (*.*)|*.*";
			d.DefaultExt = "dll";
			if (d.ShowDialog() == DialogResult.OK) AssemblyFile.Text = d.FileName;		
		}

		private void Start_Click(object sender, System.EventArgs e)
		{
			StopThread = false;
			AsyncCallback cb = new AsyncCallback(this.LinkResult);
			LinkCallback lcb = new LinkCallback(this.Link);
			ControlEnabler(true);
			lcb.BeginInvoke(cb, this);
		}

		private void Link()
		{
			try
			{
				Ret = null;
				MyConfig.TopSlopeMultiplier = Convert.ToDouble(TopSlopeMultiplier.Text);
				MyConfig.BottomSlopeMultiplier = Convert.ToDouble(BottomSlopeMultiplier.Text);
				MyConfig.TopSlopeDX= Convert.ToDouble(TopSlopeDX.Text);
				MyConfig.BottomSlopeDX = Convert.ToDouble(BottomSlopeDX.Text);
				MyConfig.TopSlopeDY= Convert.ToDouble(TopSlopeDY.Text);
				MyConfig.BottomSlopeDY = Convert.ToDouble(BottomSlopeDY.Text);
				if (ifl == null || imng == null) LoadAssemblyModule();
				System.IO.FileStream f = new System.IO.FileStream(CatalogFile.Text, System.IO.FileMode.Open, System.IO.FileAccess.Read);
				Catalog Cat = new Catalog(f);
				f.Close();
				Ret = ifl.Link(Cat, CHORUSFormat.Checked ? typeof(SySal.Scanning.Plate.IO.CHORUS.LinkedZone) : typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone));
				System.IO.FileStream o = new System.IO.FileStream(OutputFile.Text, System.IO.FileMode.Create);
				Ret.Save(o);
				o.Flush();
				o.Close();
			}
			catch (Exception x)
			{
				System.Windows.Forms.MessageBox.Show(x.ToString(), x.Message);
			}
		}

		private void Stop_Click(object sender, System.EventArgs e)
		{
			StopThread = true;
		}

		private void LoadAssembly_Click(object sender, System.EventArgs e)
		{
			LoadAssemblyModule();
		}

		private void cmdComputeShrink_Click(object sender, System.EventArgs e)
		{
			try
			{
				EasyLinkNET.EasyShrink esh = new EasyLinkNET.EasyShrink(Ret, 
					Convert.ToDouble(this.TopSlopeMultiplier.Text) , Convert.ToDouble(this.BottomSlopeMultiplier.Text),
					Convert.ToDouble(this.TopSlopeDX.Text) , Convert.ToDouble(this.BottomSlopeDX.Text),
					Convert.ToDouble(this.TopSlopeDY.Text) , Convert.ToDouble(this.BottomSlopeDY.Text));
				esh.Show();
			}
			catch
			{
			}
		}
	}

	class MIPEmulsionTrack : SySal.Scanning.MIPIndexedEmulsionTrack
	{
		public static void AdjustSlopes(SySal.Scanning.MIPIndexedEmulsionTrack t, double slopemultiplier, double slopedx, double slopedy)
		{
			SySal.Tracking.MIPEmulsionTrackInfo info = MIPEmulsionTrack.AccessInfo(t);
			info.Slope.X = info.Slope.X * slopemultiplier + slopedx;
			info.Slope.Y = info.Slope.Y * slopemultiplier + slopedy;
		}
	}

	internal delegate void LinkCallback();
}
