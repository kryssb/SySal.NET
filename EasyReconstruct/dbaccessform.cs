using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.EasyReconstruct
{
	/// <summary>
	/// DBAccessForm - allows the user to set his/her access credentials to the DB.
	/// </summary>
	public class DBAccessForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox txtDBName;
		private System.Windows.Forms.TextBox txtUserName;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox txtPassword;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Button cmdOK;
		private System.Windows.Forms.Button cmdCancel;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		internal string dbname;
		internal string userid;
		internal string pwd;

		internal DBAccessForm()
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
			this.label1 = new System.Windows.Forms.Label();
			this.txtDBName = new System.Windows.Forms.TextBox();
			this.txtUserName = new System.Windows.Forms.TextBox();
			this.label2 = new System.Windows.Forms.Label();
			this.txtPassword = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.cmdOK = new System.Windows.Forms.Button();
			this.cmdCancel = new System.Windows.Forms.Button();
			this.SuspendLayout();
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point(8, 16);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(89, 13);
			this.label1.TabIndex = 0;
			this.label1.Text = "Database Name:";
			// 
			// txtDBName
			// 
			this.txtDBName.Location = new System.Drawing.Point(96, 14);
			this.txtDBName.Name = "txtDBName";
			this.txtDBName.Size = new System.Drawing.Size(176, 20);
			this.txtDBName.TabIndex = 1;
			this.txtDBName.Text = "";
			// 
			// txtUserName
			// 
			this.txtUserName.Location = new System.Drawing.Point(96, 48);
			this.txtUserName.Name = "txtUserName";
			this.txtUserName.Size = new System.Drawing.Size(176, 20);
			this.txtUserName.TabIndex = 3;
			this.txtUserName.Text = "";
			// 
			// label2
			// 
			this.label2.AutoSize = true;
			this.label2.Location = new System.Drawing.Point(8, 48);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(64, 13);
			this.label2.TabIndex = 2;
			this.label2.Text = "User Name:";
			// 
			// txtPassword
			// 
			this.txtPassword.Location = new System.Drawing.Point(96, 80);
			this.txtPassword.Name = "txtPassword";
			this.txtPassword.PasswordChar = '*';
			this.txtPassword.Size = new System.Drawing.Size(176, 20);
			this.txtPassword.TabIndex = 5;
			this.txtPassword.Text = "";
			// 
			// label3
			// 
			this.label3.AutoSize = true;
			this.label3.Location = new System.Drawing.Point(8, 80);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(57, 13);
			this.label3.TabIndex = 4;
			this.label3.Text = "Password:";
			// 
			// cmdOK
			// 
			this.cmdOK.DialogResult = System.Windows.Forms.DialogResult.OK;
			this.cmdOK.Location = new System.Drawing.Point(24, 120);
			this.cmdOK.Name = "cmdOK";
			this.cmdOK.Size = new System.Drawing.Size(88, 24);
			this.cmdOK.TabIndex = 6;
			this.cmdOK.Text = "OK";
			this.cmdOK.Click += new System.EventHandler(this.cmdOK_Click);
			// 
			// cmdCancel
			// 
			this.cmdCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			this.cmdCancel.Location = new System.Drawing.Point(152, 120);
			this.cmdCancel.Name = "cmdCancel";
			this.cmdCancel.Size = new System.Drawing.Size(96, 24);
			this.cmdCancel.TabIndex = 7;
			this.cmdCancel.Text = "Cancel";
			this.cmdCancel.Click += new System.EventHandler(this.cmdCancel_Click);
			// 
			// DBAccessForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(280, 158);
			this.Controls.AddRange(new System.Windows.Forms.Control[] {
																		  this.cmdCancel,
																		  this.cmdOK,
																		  this.txtPassword,
																		  this.label3,
																		  this.txtUserName,
																		  this.label2,
																		  this.txtDBName,
																		  this.label1});
			this.MaximizeBox = false;
			this.Name = "DBAccessForm";
			this.Text = "Database Login";
			this.Load += new System.EventHandler(this.DBAccessForm_Load);
			this.ResumeLayout(false);

		}
		#endregion

		private void DBAccessForm_Load(object sender, System.EventArgs e)
		{

		}

		public void feed(string dbname)
		{
			this.txtDBName.Text = dbname;
		}

		private void cmdOK_Click(object sender, System.EventArgs e)
		{
		
			try
			{
				if (userid=="" || pwd == "" || this.txtDBName.Text == "") throw new Exception("All fields must be filled");
				dbname = this.txtDBName.Text;
				userid = this.txtUserName.Text;
				pwd = this.txtPassword.Text;
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
