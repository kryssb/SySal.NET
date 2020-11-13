using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.QuickDataCheck
{
	/// <summary>
	/// Database access form.
	/// </summary>
	/// <remarks>
	/// <para>This form is preloaded with the credentials from the default credential record.</para>
	/// <para>The user can change these credentials to access other DBs or with other user credentials.</para>
	/// </remarks>
	public class DBAccessForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Button cmdCancel;
		private System.Windows.Forms.Button cmdOK;
		private System.Windows.Forms.TextBox txtPassword;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.TextBox txtUserName;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox txtDBName;
		private System.Windows.Forms.Label label1;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		public SySal.OperaDb.OperaDbCredentials newDBCred;

		public DBAccessForm()
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
			this.cmdCancel = new System.Windows.Forms.Button();
			this.cmdOK = new System.Windows.Forms.Button();
			this.txtPassword = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.txtUserName = new System.Windows.Forms.TextBox();
			this.label2 = new System.Windows.Forms.Label();
			this.txtDBName = new System.Windows.Forms.TextBox();
			this.label1 = new System.Windows.Forms.Label();
			this.SuspendLayout();
			// 
			// cmdCancel
			// 
			this.cmdCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			this.cmdCancel.Location = new System.Drawing.Point(158, 120);
			this.cmdCancel.Name = "cmdCancel";
			this.cmdCancel.Size = new System.Drawing.Size(96, 24);
			this.cmdCancel.TabIndex = 15;
			this.cmdCancel.Text = "Cancel";
			this.cmdCancel.Click += new System.EventHandler(this.cmdCancel_Click);
			// 
			// cmdOK
			// 
			this.cmdOK.DialogResult = System.Windows.Forms.DialogResult.OK;
			this.cmdOK.Location = new System.Drawing.Point(30, 120);
			this.cmdOK.Name = "cmdOK";
			this.cmdOK.Size = new System.Drawing.Size(88, 24);
			this.cmdOK.TabIndex = 14;
			this.cmdOK.Text = "OK";
			this.cmdOK.Click += new System.EventHandler(this.cmdOK_Click);
			// 
			// txtPassword
			// 
			this.txtPassword.Location = new System.Drawing.Point(102, 80);
			this.txtPassword.Name = "txtPassword";
			this.txtPassword.PasswordChar = '*';
			this.txtPassword.Size = new System.Drawing.Size(176, 20);
			this.txtPassword.TabIndex = 13;
			this.txtPassword.Text = "";
			// 
			// label3
			// 
			this.label3.AutoSize = true;
			this.label3.Location = new System.Drawing.Point(14, 80);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(57, 13);
			this.label3.TabIndex = 12;
			this.label3.Text = "Password:";
			// 
			// txtUserName
			// 
			this.txtUserName.Location = new System.Drawing.Point(102, 48);
			this.txtUserName.Name = "txtUserName";
			this.txtUserName.Size = new System.Drawing.Size(176, 20);
			this.txtUserName.TabIndex = 11;
			this.txtUserName.Text = "";
			// 
			// label2
			// 
			this.label2.AutoSize = true;
			this.label2.Location = new System.Drawing.Point(14, 48);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(64, 13);
			this.label2.TabIndex = 10;
			this.label2.Text = "User Name:";
			// 
			// txtDBName
			// 
			this.txtDBName.Location = new System.Drawing.Point(102, 8);
			this.txtDBName.Name = "txtDBName";
			this.txtDBName.Size = new System.Drawing.Size(176, 20);
			this.txtDBName.TabIndex = 9;
			this.txtDBName.Text = "";
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point(14, 16);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(89, 13);
			this.label1.TabIndex = 8;
			this.label1.Text = "Database Name:";
			// 
			// DBAccessForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(288, 150);
			this.Controls.AddRange(new System.Windows.Forms.Control[] {
																		  this.cmdCancel,
																		  this.cmdOK,
																		  this.txtPassword,
																		  this.label3,
																		  this.label2,
																		  this.label1,
																		  this.txtUserName,
																		  this.txtDBName});
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "DBAccessForm";
			this.Text = "DB connection";
			this.ResumeLayout(false);

		}
		#endregion

		public void feed(SySal.OperaDb.OperaDbCredentials cred)
		{
			this.txtDBName.Text = cred.DBServer;
			this.txtUserName.Text = cred.DBUserName;
			this.txtPassword.Text = cred.DBPassword;
		}

		private void cmdOK_Click(object sender, System.EventArgs e)
		{
			try
			{
//				if (userid=="" || pwd == "" || this.txtDBName.Text == "") throw new Exception("All fields must be filled");
				newDBCred = new SySal.OperaDb.OperaDbCredentials();
				newDBCred.DBServer = this.txtDBName.Text;
				newDBCred.DBUserName = this.txtUserName.Text;
				newDBCred.DBPassword = this.txtPassword.Text;
				Close();
			} 
			catch
			{

			}

		}

		private void cmdCancel_Click(object sender, System.EventArgs e)
		{
			Close();
		}
	}
}
