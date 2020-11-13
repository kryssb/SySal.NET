using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.SftViewer
{
	/// <summary>
	/// Summary description for frmLogin.
	/// </summary>
	public class frmLogin : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Button btnOK;
		private System.Windows.Forms.Button btnCancel;
		private System.Windows.Forms.TextBox txtDbServer;
		private System.Windows.Forms.TextBox txtUsername;
		private System.Windows.Forms.TextBox txtPassword;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public frmLogin()
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
				if(components != null)
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
			this.txtDbServer = new System.Windows.Forms.TextBox();
			this.label1 = new System.Windows.Forms.Label();
			this.label2 = new System.Windows.Forms.Label();
			this.txtUsername = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.txtPassword = new System.Windows.Forms.TextBox();
			this.btnOK = new System.Windows.Forms.Button();
			this.btnCancel = new System.Windows.Forms.Button();
			this.SuspendLayout();
			// 
			// txtDbServer
			// 
			this.txtDbServer.Location = new System.Drawing.Point(88, 16);
			this.txtDbServer.Name = "txtDbServer";
			this.txtDbServer.Size = new System.Drawing.Size(224, 20);
			this.txtDbServer.TabIndex = 0;
			this.txtDbServer.Text = "";
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(16, 16);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(64, 23);
			this.label1.TabIndex = 1;
			this.label1.Text = "DB server";
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(16, 40);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(64, 23);
			this.label2.TabIndex = 3;
			this.label2.Text = "Username";
			// 
			// txtUsername
			// 
			this.txtUsername.Location = new System.Drawing.Point(88, 40);
			this.txtUsername.Name = "txtUsername";
			this.txtUsername.Size = new System.Drawing.Size(224, 20);
			this.txtUsername.TabIndex = 2;
			this.txtUsername.Text = "";
			// 
			// label3
			// 
			this.label3.Location = new System.Drawing.Point(16, 64);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(64, 23);
			this.label3.TabIndex = 5;
			this.label3.Text = "Password";
			// 
			// txtPassword
			// 
			this.txtPassword.Location = new System.Drawing.Point(88, 64);
			this.txtPassword.Name = "txtPassword";
			this.txtPassword.PasswordChar = '*';
			this.txtPassword.Size = new System.Drawing.Size(224, 20);
			this.txtPassword.TabIndex = 4;
			this.txtPassword.Text = "";
			// 
			// btnOK
			// 
			this.btnOK.DialogResult = System.Windows.Forms.DialogResult.OK;
			this.btnOK.Location = new System.Drawing.Point(152, 96);
			this.btnOK.Name = "btnOK";
			this.btnOK.TabIndex = 6;
			this.btnOK.Text = "OK";
			this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
			// 
			// btnCancel
			// 
			this.btnCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			this.btnCancel.Location = new System.Drawing.Point(232, 96);
			this.btnCancel.Name = "btnCancel";
			this.btnCancel.TabIndex = 7;
			this.btnCancel.Text = "Cancel";
			this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
			// 
			// frmLogin
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(320, 126);
			this.Controls.Add(this.btnCancel);
			this.Controls.Add(this.btnOK);
			this.Controls.Add(this.label3);
			this.Controls.Add(this.txtPassword);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.txtUsername);
			this.Controls.Add(this.label1);
			this.Controls.Add(this.txtDbServer);
			this.MaximumSize = new System.Drawing.Size(328, 160);
			this.MinimizeBox = false;
			this.MinimumSize = new System.Drawing.Size(328, 160);
			this.Name = "frmLogin";
			this.ShowInTaskbar = false;
			this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
			this.Text = "Database login";
			this.Load += new System.EventHandler(this.frmLogin_Load);
			this.ResumeLayout(false);

		}
		#endregion

		/*private void label3_Click(object sender, System.EventArgs e)
		{
		
		}

		private void textBox3_TextChanged(object sender, System.EventArgs e)
		{
		
		}*/

		private void btnCancel_Click(object sender, System.EventArgs e)
		{
			Close();
		}

		private void btnOK_Click(object sender, System.EventArgs e)
		{
			try
			{
				SySal.OperaDb.OperaDbCredentials cred = new SySal.OperaDb.OperaDbCredentials();
				cred.DBPassword = txtPassword.Text.Trim();
				cred.DBServer = txtDbServer.Text.Trim();
				cred.DBUserName = txtUsername.Text.Trim();
				frmMain.Conn = cred.Connect();
				frmMain.Conn.Open();
				SySal.OperaDb.Schema.DB = frmMain.Conn;
				/*frmMain.Settings.UserName = cred.DBUserName;
				frmMain.Settings.Password = cred.DBPassword;
				frmMain.Settings.DbServer = cred.DBServer;*/
				
				
			}
			catch (Exception ex)
			{
				MessageBox.Show(ex.ToString());
				this.DialogResult = System.Windows.Forms.DialogResult.Cancel;
				//btnCancel_Click(sender, e);
			}
			
			Close();
		}

		private void frmLogin_Load(object sender, System.EventArgs e)
		{
			SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
			txtPassword.Text = cred.DBPassword;
			txtUsername.Text = cred.DBUserName;
			txtDbServer.Text = cred.DBServer;
			/*if (frmMain.Settings != null)
			{
				txtPassword.Text = frmMain.Settings.Password;
				txtUsername.Text = frmMain.Settings.UserName;
				txtDbServer.Text = frmMain.Settings.DbServer;
			}*/
		}
	}
}
