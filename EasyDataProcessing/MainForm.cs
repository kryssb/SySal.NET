using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;

namespace SySal.Executables.EasyDataProcessing
{
	/// <summary>
	/// EasyDataProcessing is a GUI tool to monitor DataProcessingServers and to schedule computations on servers and clusters of servers.
	/// </summary>
	/// <remarks>
	/// <para>If the Manager flag is selected, the port for the manager of a cluster of DataProcessingServers is queried. If the flag is deselected, a stand-alone DataProcessingServer is queried.</para>
	/// <para>The Test button checks the currently available processing power.</para>
	/// <para>The Enqueue button queues a new batch to the list of batches of the specified DataProcessingServer. It does not wait for completion.</para>
	/// <para>The EnqueueAndWait button queues a new batch and waits for its completion. The application is marked as "not responding" by the OS until the batch is complete.</para>
	/// <para>The Remove button attempts to remove a batch from the execution list of the specified DataProcessingServer.</para>
	/// <para>The Queue button queues a list of batches to the specified DataProcessingServer.</para>
	/// <para>
	/// The syntax to specify a batch list for execution is the following:
	/// <example>
	/// <code>
	/// &lt;BatchList&gt;
	///  &lt;Batch&gt;
	///   &lt;Filename&gt;here goes the full path to the executable to be launched&lt;/Filename&gt;
	///   &lt;CommandLineArguments&gt;here go the command line arguments, if any; if this field is missing, no arguments will be passed.&lt;/CommandLineArguments&gt;
	///   &lt;Description&gt;the description of the batch.&lt;/Description&gt;
	///   &lt;MachinePowerClass&gt;a nonnegative number specifying the minimum power class of the machines to use for this batch.&lt;/MachinePowerClass&gt;
	///  &lt;/Batch&gt;
	///  &lt;Batch&gt;
	///   ...
	///  &lt;/Batch&gt;
	///  
	///  ...
	///  
	/// &lt;/BatchList&gt;
	/// </code>
	/// </example>
	/// </para>
	/// <para>By double-clicking on a batch in the queue of the specified DataProcessingServer, one can learn the associated information (owner, arguments, etc.).</para>
	/// </remarks>
	public class MainForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Button TestButton;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox DataProcText;
		private System.Windows.Forms.Button ViewQueueButton;
		private System.Windows.Forms.Button EnqueueAndWaitButton;
		private System.Windows.Forms.Button EnqueueButton;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox UsernameText;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.TextBox DescriptionText;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.ListView QueueList;
		private System.Windows.Forms.ColumnHeader columnHeader1;
		private System.Windows.Forms.ColumnHeader columnHeader2;
		private System.Windows.Forms.ColumnHeader columnHeader3;
		private System.Windows.Forms.TextBox PasswordText;
		private System.Windows.Forms.Button RemoveButton;
		private System.Windows.Forms.Label PowerClassText;
		private System.Windows.Forms.Button ExePathButton;
		private System.Windows.Forms.CheckBox BatchManagerCheck;
		private System.Windows.Forms.TextBox ExePathText;
		private System.Windows.Forms.TextBox ExeOutputsText;
		private System.Windows.Forms.TextBox ArgumentsText;
		private System.Windows.Forms.Button QueueBatchList;
		private System.Windows.Forms.TextBox AliasPasswordText;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.TextBox AliasUsernameText;
		private System.Windows.Forms.Label label7;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public MainForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			SySal.OperaDb.OperaDbCredentials oc = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
			UsernameText.Text = oc.OPERAUserName;
			PasswordText.Text = oc.OPERAPassword;
			AliasUsernameText.Text = oc.DBUserName;
			AliasPasswordText.Text = oc.DBPassword;
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainForm));
            this.TestButton = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.DataProcText = new System.Windows.Forms.TextBox();
            this.ViewQueueButton = new System.Windows.Forms.Button();
            this.QueueList = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.EnqueueAndWaitButton = new System.Windows.Forms.Button();
            this.EnqueueButton = new System.Windows.Forms.Button();
            this.label2 = new System.Windows.Forms.Label();
            this.UsernameText = new System.Windows.Forms.TextBox();
            this.PasswordText = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.ExePathText = new System.Windows.Forms.TextBox();
            this.ArgumentsText = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.DescriptionText = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.ExeOutputsText = new System.Windows.Forms.TextBox();
            this.ExePathButton = new System.Windows.Forms.Button();
            this.RemoveButton = new System.Windows.Forms.Button();
            this.PowerClassText = new System.Windows.Forms.Label();
            this.BatchManagerCheck = new System.Windows.Forms.CheckBox();
            this.QueueBatchList = new System.Windows.Forms.Button();
            this.AliasPasswordText = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.AliasUsernameText = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.groupBox1.SuspendLayout();
            this.SuspendLayout();
            // 
            // TestButton
            // 
            this.TestButton.Location = new System.Drawing.Point(8, 32);
            this.TestButton.Name = "TestButton";
            this.TestButton.Size = new System.Drawing.Size(64, 24);
            this.TestButton.TabIndex = 0;
            this.TestButton.Text = "Test";
            this.TestButton.Click += new System.EventHandler(this.TestButton_Click);
            // 
            // label1
            // 
            this.label1.Location = new System.Drawing.Point(8, 8);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(72, 24);
            this.label1.TabIndex = 22;
            this.label1.Text = "Server";
            // 
            // DataProcText
            // 
            this.DataProcText.Location = new System.Drawing.Point(80, 8);
            this.DataProcText.Name = "DataProcText";
            this.DataProcText.Size = new System.Drawing.Size(264, 20);
            this.DataProcText.TabIndex = 1;
            // 
            // ViewQueueButton
            // 
            this.ViewQueueButton.Location = new System.Drawing.Point(192, 32);
            this.ViewQueueButton.Name = "ViewQueueButton";
            this.ViewQueueButton.Size = new System.Drawing.Size(104, 24);
            this.ViewQueueButton.TabIndex = 3;
            this.ViewQueueButton.Text = "View queue";
            this.ViewQueueButton.Click += new System.EventHandler(this.ViewQueueButton_Click);
            // 
            // QueueList
            // 
            this.QueueList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2,
            this.columnHeader3});
            this.QueueList.FullRowSelect = true;
            this.QueueList.GridLines = true;
            this.QueueList.Location = new System.Drawing.Point(8, 64);
            this.QueueList.MultiSelect = false;
            this.QueueList.Name = "QueueList";
            this.QueueList.Size = new System.Drawing.Size(520, 192);
            this.QueueList.TabIndex = 4;
            this.QueueList.UseCompatibleStateImageBehavior = false;
            this.QueueList.View = System.Windows.Forms.View.Details;
            this.QueueList.DoubleClick += new System.EventHandler(this.QueueList_DoubleClick);
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Id";
            this.columnHeader1.Width = 130;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Description";
            this.columnHeader2.Width = 310;
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "User";
            // 
            // EnqueueAndWaitButton
            // 
            this.EnqueueAndWaitButton.Location = new System.Drawing.Point(88, 264);
            this.EnqueueAndWaitButton.Name = "EnqueueAndWaitButton";
            this.EnqueueAndWaitButton.Size = new System.Drawing.Size(128, 24);
            this.EnqueueAndWaitButton.TabIndex = 6;
            this.EnqueueAndWaitButton.Text = "Enqueue and wait";
            this.EnqueueAndWaitButton.Click += new System.EventHandler(this.EnqueueAndWaitButton_Click);
            // 
            // EnqueueButton
            // 
            this.EnqueueButton.Location = new System.Drawing.Point(8, 264);
            this.EnqueueButton.Name = "EnqueueButton";
            this.EnqueueButton.Size = new System.Drawing.Size(72, 24);
            this.EnqueueButton.TabIndex = 5;
            this.EnqueueButton.Text = "Enqueue";
            this.EnqueueButton.Click += new System.EventHandler(this.EnqueueButton_Click);
            // 
            // label2
            // 
            this.label2.Location = new System.Drawing.Point(8, 320);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(72, 24);
            this.label2.TabIndex = 9;
            this.label2.Text = "Username";
            // 
            // UsernameText
            // 
            this.UsernameText.Location = new System.Drawing.Point(128, 320);
            this.UsernameText.Name = "UsernameText";
            this.UsernameText.Size = new System.Drawing.Size(104, 20);
            this.UsernameText.TabIndex = 10;
            // 
            // PasswordText
            // 
            this.PasswordText.Location = new System.Drawing.Point(368, 320);
            this.PasswordText.Name = "PasswordText";
            this.PasswordText.PasswordChar = '*';
            this.PasswordText.Size = new System.Drawing.Size(104, 20);
            this.PasswordText.TabIndex = 12;
            // 
            // label3
            // 
            this.label3.Location = new System.Drawing.Point(256, 320);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(72, 24);
            this.label3.TabIndex = 11;
            this.label3.Text = "Password";
            // 
            // ExePathText
            // 
            this.ExePathText.Location = new System.Drawing.Point(88, 368);
            this.ExePathText.Name = "ExePathText";
            this.ExePathText.Size = new System.Drawing.Size(440, 20);
            this.ExePathText.TabIndex = 14;
            // 
            // ArgumentsText
            // 
            this.ArgumentsText.Location = new System.Drawing.Point(88, 392);
            this.ArgumentsText.Name = "ArgumentsText";
            this.ArgumentsText.Size = new System.Drawing.Size(440, 20);
            this.ArgumentsText.TabIndex = 16;
            // 
            // label5
            // 
            this.label5.ImageAlign = System.Drawing.ContentAlignment.TopLeft;
            this.label5.Location = new System.Drawing.Point(8, 392);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(72, 24);
            this.label5.TabIndex = 15;
            this.label5.Text = "Arguments";
            this.label5.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // DescriptionText
            // 
            this.DescriptionText.Location = new System.Drawing.Point(128, 296);
            this.DescriptionText.Name = "DescriptionText";
            this.DescriptionText.Size = new System.Drawing.Size(400, 20);
            this.DescriptionText.TabIndex = 8;
            // 
            // label6
            // 
            this.label6.Location = new System.Drawing.Point(8, 296);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(72, 24);
            this.label6.TabIndex = 7;
            this.label6.Text = "Description";
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.ExeOutputsText);
            this.groupBox1.Location = new System.Drawing.Point(536, 0);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(360, 416);
            this.groupBox1.TabIndex = 19;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Execution outputs";
            // 
            // ExeOutputsText
            // 
            this.ExeOutputsText.Location = new System.Drawing.Point(8, 16);
            this.ExeOutputsText.Multiline = true;
            this.ExeOutputsText.Name = "ExeOutputsText";
            this.ExeOutputsText.ScrollBars = System.Windows.Forms.ScrollBars.Both;
            this.ExeOutputsText.Size = new System.Drawing.Size(344, 392);
            this.ExeOutputsText.TabIndex = 21;
            // 
            // ExePathButton
            // 
            this.ExePathButton.Location = new System.Drawing.Point(8, 368);
            this.ExePathButton.Name = "ExePathButton";
            this.ExePathButton.Size = new System.Drawing.Size(72, 24);
            this.ExePathButton.TabIndex = 20;
            this.ExePathButton.Text = "Exe path";
            this.ExePathButton.Click += new System.EventHandler(this.ExePathButton_Click);
            // 
            // RemoveButton
            // 
            this.RemoveButton.Location = new System.Drawing.Point(224, 264);
            this.RemoveButton.Name = "RemoveButton";
            this.RemoveButton.Size = new System.Drawing.Size(72, 24);
            this.RemoveButton.TabIndex = 20;
            this.RemoveButton.Text = "Remove";
            this.RemoveButton.Click += new System.EventHandler(this.RemoveButton_Click);
            // 
            // PowerClassText
            // 
            this.PowerClassText.Location = new System.Drawing.Point(80, 32);
            this.PowerClassText.Name = "PowerClassText";
            this.PowerClassText.Size = new System.Drawing.Size(104, 24);
            this.PowerClassText.TabIndex = 21;
            this.PowerClassText.Text = "Power Class: -";
            this.PowerClassText.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // BatchManagerCheck
            // 
            this.BatchManagerCheck.Checked = true;
            this.BatchManagerCheck.CheckState = System.Windows.Forms.CheckState.Checked;
            this.BatchManagerCheck.Location = new System.Drawing.Point(360, 8);
            this.BatchManagerCheck.Name = "BatchManagerCheck";
            this.BatchManagerCheck.Size = new System.Drawing.Size(80, 24);
            this.BatchManagerCheck.TabIndex = 2;
            this.BatchManagerCheck.Text = "Manager";
            // 
            // QueueBatchList
            // 
            this.QueueBatchList.Location = new System.Drawing.Point(400, 264);
            this.QueueBatchList.Name = "QueueBatchList";
            this.QueueBatchList.Size = new System.Drawing.Size(128, 24);
            this.QueueBatchList.TabIndex = 23;
            this.QueueBatchList.Text = "Queue batch list";
            this.QueueBatchList.Click += new System.EventHandler(this.QueueBatchList_Click);
            // 
            // AliasPasswordText
            // 
            this.AliasPasswordText.Location = new System.Drawing.Point(368, 344);
            this.AliasPasswordText.Name = "AliasPasswordText";
            this.AliasPasswordText.PasswordChar = '*';
            this.AliasPasswordText.Size = new System.Drawing.Size(104, 20);
            this.AliasPasswordText.TabIndex = 27;
            // 
            // label4
            // 
            this.label4.Location = new System.Drawing.Point(256, 344);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(104, 24);
            this.label4.TabIndex = 26;
            this.label4.Text = "Alias Password";
            // 
            // AliasUsernameText
            // 
            this.AliasUsernameText.Location = new System.Drawing.Point(128, 344);
            this.AliasUsernameText.Name = "AliasUsernameText";
            this.AliasUsernameText.Size = new System.Drawing.Size(104, 20);
            this.AliasUsernameText.TabIndex = 25;
            // 
            // label7
            // 
            this.label7.Location = new System.Drawing.Point(8, 344);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(112, 24);
            this.label7.TabIndex = 24;
            this.label7.Text = "Alias Username";
            // 
            // MainForm
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(904, 423);
            this.Controls.Add(this.AliasPasswordText);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.AliasUsernameText);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.QueueBatchList);
            this.Controls.Add(this.BatchManagerCheck);
            this.Controls.Add(this.PowerClassText);
            this.Controls.Add(this.RemoveButton);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.DescriptionText);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.ArgumentsText);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.ExePathText);
            this.Controls.Add(this.PasswordText);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.UsernameText);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.EnqueueButton);
            this.Controls.Add(this.EnqueueAndWaitButton);
            this.Controls.Add(this.QueueList);
            this.Controls.Add(this.ViewQueueButton);
            this.Controls.Add(this.DataProcText);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.TestButton);
            this.Controls.Add(this.ExePathButton);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "MainForm";
            this.Text = "Easy Data Processing";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

		}
		#endregion

		static System.Random Rnd = new System.Random();

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{
			System.Runtime.Remoting.Channels.ChannelServices.RegisterChannel(new TcpChannel());
			Application.Run(new MainForm());
		}

		private void TestButton_Click(object sender, System.EventArgs e)
		{
			try
			{
				SySal.DAQSystem.IDataProcessingServer Srv = new SySal.DAQSystem.SyncDataProcessingServerWrapper((SySal.DAQSystem.IDataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.IDataProcessingServer), "tcp://" + DataProcText.Text + ":" + (BatchManagerCheck.Checked ? (int)SySal.DAQSystem.OperaPort.BatchServer : (int)SySal.DAQSystem.OperaPort.DataProcessingServer).ToString() + "/DataProcessingServer.rem"), System.TimeSpan.FromMilliseconds(10000));
				int commpar = Rnd.Next();
				if (Srv.TestComm(commpar) != 2 * commpar - 1) throw new Exception("The service at the specified port is not a DataProcessingServer.");
				PowerClassText.Text = "Power Class: " + Srv.MachinePowerClass;
				MessageBox.Show("Communication OK", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
			}
			catch (Exception x)
			{				
				MessageBox.Show(x.Message, "Remoting error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
			GC.Collect();
		}

		private void ViewQueueButton_Click(object sender, System.EventArgs e)
		{
			QueueList.Items.Clear();
			try
			{
				SySal.DAQSystem.IDataProcessingServer Srv = new SySal.DAQSystem.SyncDataProcessingServerWrapper((SySal.DAQSystem.IDataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.IDataProcessingServer), "tcp://" + DataProcText.Text + ":" + (BatchManagerCheck.Checked ? (int)SySal.DAQSystem.OperaPort.BatchServer : (int)SySal.DAQSystem.OperaPort.DataProcessingServer).ToString() + "/DataProcessingServer.rem"), System.TimeSpan.FromMilliseconds(10000));
				SySal.DAQSystem.DataProcessingBatchDesc [] queue = Srv.Queue;
				foreach (SySal.DAQSystem.DataProcessingBatchDesc d in queue)
				{
					ListViewItem li = QueueList.Items.Add(d.Id.ToString("X16"));
					li.SubItems.Add(d.Description);
					li.SubItems.Add(d.Username);
					li.Tag = d;
				}
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Remoting error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}		
			GC.Collect();
		}

		private void EnqueueButton_Click(object sender, System.EventArgs e)
		{
			SySal.DAQSystem.DataProcessingBatchDesc desc = new SySal.DAQSystem.DataProcessingBatchDesc();
			desc.Description = DescriptionText.Text;
			desc.AliasUsername = AliasUsernameText.Text;
			desc.AliasPassword = AliasPasswordText.Text;
			desc.Username = UsernameText.Text;
			desc.Password = PasswordText.Text;
			desc.MachinePowerClass = 0;
			desc.CommandLineArguments = ArgumentsText.Text;
			desc.Filename = ExePathText.Text;
			try
			{
				SySal.DAQSystem.IDataProcessingServer Srv = new SySal.DAQSystem.SyncDataProcessingServerWrapper((SySal.DAQSystem.IDataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.IDataProcessingServer), "tcp://" + DataProcText.Text + ":" + (BatchManagerCheck.Checked ? (int)SySal.DAQSystem.OperaPort.BatchServer : (int)SySal.DAQSystem.OperaPort.DataProcessingServer).ToString() + "/DataProcessingServer.rem"), System.TimeSpan.FromMilliseconds(10000));
				desc.Id = Srv.SuggestId;
				bool batchaccepted = Srv.Enqueue(desc);
				if (!batchaccepted)
					MessageBox.Show("Batch Rejected", "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Remoting error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
			GC.Collect();
			ViewQueueButton_Click(this, null);
		}

		private void DataProcessingComplete(SySal.DAQSystem.DataProcessingBatchDesc desc, System.Exception x)
		{
			string str = ExeOutputsText.Text;
			int percent = 0;
			try
			{
				percent = Convert.ToInt32(Math.Round(100 * desc.TotalProcessorTime.TotalMilliseconds / (desc.Finished - desc.Started).TotalMilliseconds));
			}
			catch (Exception) { percent = 0; };				
			str += "\r\n\r\nProcess " + desc.Id.ToString("X16") + "\r\nStart: " + desc.Started.ToString() + " Finish: " + desc.Finished.ToString() + 
				"\r\nCPU Time: " + desc.TotalProcessorTime.ToString() + "\r\nCPU Time / Total time: " + percent.ToString() + 
				"%\r\n" + "Peak Virtual Memory (MB): " + (desc.PeakVirtualMemorySize / 1048576).ToString() +
				"\r\nPeak Working Set (MB): " + (desc.PeakWorkingSet / 1048576).ToString();
			if (x != null)
				str += "\r\nException: " + x.GetType().ToString() + "\r\n" + x.Message + "\r\n\r\nDetails: \r\n" + x.ToString();
			ExeOutputsText.Text = str;		
		}

		private void EnqueueAndWaitButton_Click(object sender, System.EventArgs e)
		{
			SySal.DAQSystem.DataProcessingBatchDesc desc = new SySal.DAQSystem.DataProcessingBatchDesc();
			desc.Description = DescriptionText.Text;
			desc.AliasUsername = AliasUsernameText.Text;
			desc.AliasPassword = AliasPasswordText.Text;
			desc.Username = UsernameText.Text;
			desc.Password = PasswordText.Text;
			desc.MachinePowerClass = 0;
			desc.CommandLineArguments = ArgumentsText.Text;
			desc.Filename = ExePathText.Text;
			try
			{
				SySal.DAQSystem.IDataProcessingServer Srv = new SySal.DAQSystem.SyncDataProcessingServerWrapper((SySal.DAQSystem.IDataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.IDataProcessingServer), "tcp://" + DataProcText.Text + ":" + (BatchManagerCheck.Checked ? (int)SySal.DAQSystem.OperaPort.BatchServer : (int)SySal.DAQSystem.OperaPort.DataProcessingServer).ToString() + "/DataProcessingServer.rem"), System.TimeSpan.FromMilliseconds(10000));
				desc.Id = Srv.SuggestId;
				bool batchaccepted = Srv.Enqueue(desc);		
				if (!batchaccepted) MessageBox.Show("Batch Rejected", "Result", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				while (!Srv.DoneWith(desc.Id)) System.Threading.Thread.Sleep(400);
				desc = Srv.Result(desc.Id);
				DataProcessingComplete(desc, null);
			}
			catch (Exception x)
			{
				DataProcessingComplete(desc, x);
			}
			GC.Collect();			
			ViewQueueButton_Click(this, null);
		}

		private void RemoveButton_Click(object sender, System.EventArgs e)
		{
			if (QueueList.SelectedItems.Count == 1)
			{
				try
				{
					SySal.DAQSystem.IDataProcessingServer Srv = new SySal.DAQSystem.SyncDataProcessingServerWrapper((SySal.DAQSystem.IDataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.IDataProcessingServer), "tcp://" + DataProcText.Text + ":" + (BatchManagerCheck.Checked ? (int)SySal.DAQSystem.OperaPort.BatchServer : (int)SySal.DAQSystem.OperaPort.DataProcessingServer).ToString() + "/DataProcessingServer.rem"), System.TimeSpan.FromMilliseconds(10000));
					Srv.Remove(Convert.ToUInt64(QueueList.SelectedItems[0].SubItems[0].Text, 16), null, UsernameText.Text, PasswordText.Text);
				}
				catch (Exception x)
				{
					MessageBox.Show(x.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				GC.Collect();
				ViewQueueButton_Click(this, null);
			}
		}

		private void ExePathButton_Click(object sender, System.EventArgs e)
		{
			OpenFileDialog mydlg = new OpenFileDialog();
			mydlg.Title = "Select program settings file";
			mydlg.Filter = "Executable files (*.exe)|*.exe|All files (*.*)|*.*";
			if (mydlg.ShowDialog() == DialogResult.OK)
			{
				ExePathText.Text = mydlg.FileName;
			}
		}

		private void QueueBatchList_Click(object sender, System.EventArgs e)
		{
			OpenFileDialog mydlg = new OpenFileDialog();
			mydlg.Title = "Select batch list";
			mydlg.Filter = "Batch list files (*.txt)|*.txt|All files (*.*)|*.*";
			if (mydlg.ShowDialog() == DialogResult.OK)
			{
				System.IO.StreamReader r = null;
				System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(BatchList));
				SySal.DAQSystem.IDataProcessingServer Srv = null;
				BatchList bl = null;
				try
				{
					r = new System.IO.StreamReader(mydlg.FileName);
					bl = (BatchList)xmls.Deserialize(r);
					Srv = new SySal.DAQSystem.SyncDataProcessingServerWrapper((SySal.DAQSystem.IDataProcessingServer)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.IDataProcessingServer), "tcp://" + DataProcText.Text + ":" + (BatchManagerCheck.Checked ? (int)SySal.DAQSystem.OperaPort.BatchServer : (int)SySal.DAQSystem.OperaPort.DataProcessingServer).ToString() + "/DataProcessingServer.rem"), System.TimeSpan.FromMilliseconds(10000));
					foreach (Batch b in bl.List)
					{
						SySal.DAQSystem.DataProcessingBatchDesc d = new SySal.DAQSystem.DataProcessingBatchDesc();
						d.MachinePowerClass = b.MachinePowerClass;
						d.Description = b.Description;
						d.Filename = b.Filename;
						d.CommandLineArguments = b.CommandLineArguments;
						d.Username = UsernameText.Text;
						d.Password = PasswordText.Text;
						d.Id = Srv.SuggestId;
						Srv.Enqueue(d);
					}					
				}
				catch (Exception) {}
				if (r != null) 
				{
					r.Close();
					r = null;
				}
				if (Srv != null)
				{
					Srv = null;
				}
				GC.Collect();
			}		
		}

		private void QueueList_DoubleClick(object sender, System.EventArgs e)
		{
			if (QueueList.SelectedItems.Count == 1)
			{
				new BatchInfoForm((SySal.DAQSystem.DataProcessingBatchDesc)QueueList.SelectedItems[0].Tag).ShowDialog();
			}
		}
	}

	/// <summary>
	/// Class that is used to read a single batch descriptors from an XML document.
	/// </summary>
	[Serializable]
	public class Batch
	{
		/// <summary>
		/// Path to the executable file.
		/// </summary>
		public string Filename = "";
		/// <summary>
		/// Command line arguments.
		/// </summary>
		public string CommandLineArguments = "";
		/// <summary>
		/// Description of the batch.
		/// </summary>
		public string Description = "";
		/// <summary>
		/// Machine power class required for the batch.
		/// </summary>
		public uint MachinePowerClass = 0;
	}

	/// <summary>
	/// Class that is used to read a list of batch descriptors from an XML document.
	/// </summary>
	[Serializable]
	public class BatchList
	{
		/// <summary>
		/// General description of the batch.
		/// </summary>
		public string Description = "";
		/// <summary>
		/// List of the batches.
		/// </summary>
		public Batch [] List = new Batch[0];
	}
}
