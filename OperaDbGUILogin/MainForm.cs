using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;

namespace SySal.Executables.OperaDbGUILogin
{
	/// <summary>
	/// OperaDbGUILogin - GUI tool to set the default credential record of the current user.
	/// </summary>
	/// <remarks>
	/// <para>Every user on a workstation/server can have his/her own default credential record for the OPERA DB and Computing Infrastructure. 
	/// This records saves continuous login requests on the OPERA DB and on the Computing Infrastructure services.</para>
	/// <para>The record is saved in the user profile in encrypted form.</para>
	/// <para>OperaDbGUILogin does not alter the record until the OK button is pressed. Pressing Cancel will leave the record unchanged.</para>
	/// <para>Pressing the "Verify Credentials" button starts a test login. Notice that in order to verify the Computing Infrastructure credentials, which are stored in the DB, the DB login must be valid.</para>
	/// <para>More than one DB server can be specified. The server name is its TNS name. Multiple server names must be separated by commas (',').</para>
	/// </remarks>
	public class MainForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox DBServerText;
		private System.Windows.Forms.TextBox DBUsernameText;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox DBPasswordText;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.TextBox OPERAPasswordText;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.TextBox OPERAUsernameText;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.Button OKButton;
		private System.Windows.Forms.Button CancelButton;
		private System.Windows.Forms.Button VerifyButton;
        private Button DumpToButton;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		/// <summary>
		/// Creates the form.
		/// </summary>
		public MainForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
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
            this.label1 = new System.Windows.Forms.Label();
            this.DBServerText = new System.Windows.Forms.TextBox();
            this.DBUsernameText = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.DBPasswordText = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.OPERAPasswordText = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.OPERAUsernameText = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.OKButton = new System.Windows.Forms.Button();
            this.CancelButton = new System.Windows.Forms.Button();
            this.VerifyButton = new System.Windows.Forms.Button();
            this.DumpToButton = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.Location = new System.Drawing.Point(8, 8);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(120, 16);
            this.label1.TabIndex = 0;
            this.label1.Text = "Opera DB Server(s)";
            // 
            // DBServerText
            // 
            this.DBServerText.Location = new System.Drawing.Point(128, 8);
            this.DBServerText.Name = "DBServerText";
            this.DBServerText.Size = new System.Drawing.Size(336, 20);
            this.DBServerText.TabIndex = 1;
            this.DBServerText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // DBUsernameText
            // 
            this.DBUsernameText.Location = new System.Drawing.Point(328, 32);
            this.DBUsernameText.Name = "DBUsernameText";
            this.DBUsernameText.Size = new System.Drawing.Size(136, 20);
            this.DBUsernameText.TabIndex = 3;
            this.DBUsernameText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label2
            // 
            this.label2.Location = new System.Drawing.Point(8, 32);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(120, 16);
            this.label2.TabIndex = 2;
            this.label2.Text = "Opera DB Username";
            // 
            // DBPasswordText
            // 
            this.DBPasswordText.Location = new System.Drawing.Point(328, 56);
            this.DBPasswordText.Name = "DBPasswordText";
            this.DBPasswordText.PasswordChar = '*';
            this.DBPasswordText.Size = new System.Drawing.Size(136, 20);
            this.DBPasswordText.TabIndex = 5;
            this.DBPasswordText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label3
            // 
            this.label3.Location = new System.Drawing.Point(8, 56);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(120, 16);
            this.label3.TabIndex = 4;
            this.label3.Text = "Opera DB Password";
            // 
            // OPERAPasswordText
            // 
            this.OPERAPasswordText.Location = new System.Drawing.Point(328, 112);
            this.OPERAPasswordText.Name = "OPERAPasswordText";
            this.OPERAPasswordText.PasswordChar = '*';
            this.OPERAPasswordText.Size = new System.Drawing.Size(136, 20);
            this.OPERAPasswordText.TabIndex = 9;
            this.OPERAPasswordText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label4
            // 
            this.label4.Location = new System.Drawing.Point(8, 112);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(168, 16);
            this.label4.TabIndex = 8;
            this.label4.Text = "Opera Computing Password";
            // 
            // OPERAUsernameText
            // 
            this.OPERAUsernameText.Location = new System.Drawing.Point(328, 88);
            this.OPERAUsernameText.Name = "OPERAUsernameText";
            this.OPERAUsernameText.Size = new System.Drawing.Size(136, 20);
            this.OPERAUsernameText.TabIndex = 7;
            this.OPERAUsernameText.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            // 
            // label5
            // 
            this.label5.Location = new System.Drawing.Point(8, 88);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(168, 16);
            this.label5.TabIndex = 6;
            this.label5.Text = "Opera Computing Username";
            // 
            // OKButton
            // 
            this.OKButton.Location = new System.Drawing.Point(8, 144);
            this.OKButton.Name = "OKButton";
            this.OKButton.Size = new System.Drawing.Size(80, 24);
            this.OKButton.TabIndex = 10;
            this.OKButton.Text = "OK";
            this.OKButton.Click += new System.EventHandler(this.OKButton_Click);
            // 
            // CancelButton
            // 
            this.CancelButton.DialogResult = System.Windows.Forms.DialogResult.Cancel;
            this.CancelButton.Location = new System.Drawing.Point(384, 144);
            this.CancelButton.Name = "CancelButton";
            this.CancelButton.Size = new System.Drawing.Size(80, 24);
            this.CancelButton.TabIndex = 11;
            this.CancelButton.Text = "Cancel";
            this.CancelButton.Click += new System.EventHandler(this.CancelButton_Click);
            // 
            // VerifyButton
            // 
            this.VerifyButton.Location = new System.Drawing.Point(96, 144);
            this.VerifyButton.Name = "VerifyButton";
            this.VerifyButton.Size = new System.Drawing.Size(136, 24);
            this.VerifyButton.TabIndex = 12;
            this.VerifyButton.Text = "Verify Credentials";
            this.VerifyButton.Click += new System.EventHandler(this.VerifyButton_Click);
            // 
            // DumpToButton
            // 
            this.DumpToButton.Location = new System.Drawing.Point(238, 143);
            this.DumpToButton.Name = "DumpToButton";
            this.DumpToButton.Size = new System.Drawing.Size(80, 24);
            this.DumpToButton.TabIndex = 13;
            this.DumpToButton.Text = "Dump to...";
            this.DumpToButton.Click += new System.EventHandler(this.DumpToButton_Click);
            // 
            // MainForm
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(474, 179);
            this.Controls.Add(this.DumpToButton);
            this.Controls.Add(this.VerifyButton);
            this.Controls.Add(this.CancelButton);
            this.Controls.Add(this.OKButton);
            this.Controls.Add(this.OPERAPasswordText);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.OPERAUsernameText);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.DBPasswordText);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.DBUsernameText);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.DBServerText);
            this.Controls.Add(this.label1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "MainForm";
            this.Text = "Login credentials for OperaDB / OPERA";
            this.Load += new System.EventHandler(this.Init);
            this.ResumeLayout(false);
            this.PerformLayout();

		}
		#endregion

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{
			Application.Run(new MainForm());
		}

		private void OKButton_Click(object sender, System.EventArgs e)
		{
			SySal.OperaDb.OperaDbCredentials oc = new SySal.OperaDb.OperaDbCredentials();
			oc.DBServer = DBServerText.Text;
			oc.DBUserName = DBUsernameText.Text;
			oc.DBPassword = DBPasswordText.Text;
			oc.OPERAUserName = OPERAUsernameText.Text;
			oc.OPERAPassword = OPERAPasswordText.Text;
			try
			{
				oc.Record();
				Close();
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Credentials could not be recorded", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
		}

		private void VerifyButton_Click(object sender, System.EventArgs e)
		{
			SySal.OperaDb.OperaDbCredentials oc = new SySal.OperaDb.OperaDbCredentials();
			oc.DBServer = DBServerText.Text;
			oc.DBUserName = DBUsernameText.Text;
			oc.DBPassword = DBPasswordText.Text;
			oc.OPERAUserName = OPERAUsernameText.Text;
			oc.OPERAPassword = OPERAPasswordText.Text;
			try
			{
				oc.CheckDbAccess();
				MessageBox.Show("OK", "Verification succeeded", MessageBoxButtons.OK, MessageBoxIcon.Information);
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Verification failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
		}

		private void Init(object sender, System.EventArgs e)
		{
			SySal.OperaDb.OperaDbCredentials oc = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
			DBServerText.Text = oc.DBServer;
			DBUsernameText.Text = oc.DBUserName;
			DBPasswordText.Text = oc.DBPassword;
			OPERAUsernameText.Text = oc.OPERAUserName;
			OPERAPasswordText.Text = oc.OPERAPassword;
		}

		private void CancelButton_Click(object sender, System.EventArgs e)
		{
			Close();		
		}

        static SaveFileDialog sdlg = new SaveFileDialog();

        private void DumpToButton_Click(object sender, EventArgs e)
        {
            sdlg.Title = "Select for encrypted login information";
            if (sdlg.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    System.IO.File.WriteAllText(sdlg.FileName, "DBServer-> " +
                        OperaDb.OperaDbCredentials.Encode(DBServerText.Text) + "\r\nDBUser-> " + OperaDb.OperaDbCredentials.Encode(DBUsernameText.Text) +
                        "\r\nDBPwd-> " + OperaDb.OperaDbCredentials.Encode(DBPasswordText.Text) + "\r\nOPERAUser-> " + OperaDb.OperaDbCredentials.Encode(OPERAUsernameText.Text) +
                        "\r\nOPERAPwd-> " + OperaDb.OperaDbCredentials.Encode(OPERAPasswordText.Text));
                    MessageBox.Show("File generated", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }
	}
}
