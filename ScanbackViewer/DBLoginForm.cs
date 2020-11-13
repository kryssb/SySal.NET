using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.ScanbackViewer
{
	/// <summary>
	/// Database login form.
	/// </summary>
	/// <remarks>
	/// <para>This form is initially preloaded with the credentials read from the default credential record.</para>
	/// <para>The user can change them to connect to other DBs or to change the user account.</para>
	/// </remarks>
	public class DBLoginForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Label label1;
		internal System.Windows.Forms.TextBox textDB;
		internal System.Windows.Forms.TextBox textUser;
		private System.Windows.Forms.Label label2;
		internal System.Windows.Forms.TextBox textPwd;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Button buttonOK;
		private System.Windows.Forms.Button buttonCancel;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public DBLoginForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
			textDB.Text = cred.DBServer;
			textUser.Text = cred.DBUserName;
			textPwd.Text = cred.DBPassword;
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
			this.textDB = new System.Windows.Forms.TextBox();
			this.textUser = new System.Windows.Forms.TextBox();
			this.label2 = new System.Windows.Forms.Label();
			this.textPwd = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.buttonOK = new System.Windows.Forms.Button();
			this.buttonCancel = new System.Windows.Forms.Button();
			this.SuspendLayout();
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(8, 8);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(72, 24);
			this.label1.TabIndex = 0;
			this.label1.Text = "DB Server";
			this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// textDB
			// 
			this.textDB.Location = new System.Drawing.Point(80, 8);
			this.textDB.Name = "textDB";
			this.textDB.Size = new System.Drawing.Size(200, 20);
			this.textDB.TabIndex = 1;
			this.textDB.Text = "";
			// 
			// textUser
			// 
			this.textUser.Location = new System.Drawing.Point(80, 40);
			this.textUser.Name = "textUser";
			this.textUser.Size = new System.Drawing.Size(200, 20);
			this.textUser.TabIndex = 3;
			this.textUser.Text = "";
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(8, 40);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(72, 24);
			this.label2.TabIndex = 2;
			this.label2.Text = "Username";
			this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// textPwd
			// 
			this.textPwd.Location = new System.Drawing.Point(80, 72);
			this.textPwd.Name = "textPwd";
			this.textPwd.PasswordChar = '*';
			this.textPwd.Size = new System.Drawing.Size(200, 20);
			this.textPwd.TabIndex = 5;
			this.textPwd.Text = "";
			// 
			// label3
			// 
			this.label3.Location = new System.Drawing.Point(8, 72);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(72, 24);
			this.label3.TabIndex = 4;
			this.label3.Text = "Password";
			this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// buttonOK
			// 
			this.buttonOK.Location = new System.Drawing.Point(224, 104);
			this.buttonOK.Name = "buttonOK";
			this.buttonOK.Size = new System.Drawing.Size(56, 24);
			this.buttonOK.TabIndex = 6;
			this.buttonOK.Text = "OK";
			this.buttonOK.Click += new System.EventHandler(this.buttonOK_Click);
			// 
			// buttonCancel
			// 
			this.buttonCancel.Location = new System.Drawing.Point(8, 104);
			this.buttonCancel.Name = "buttonCancel";
			this.buttonCancel.Size = new System.Drawing.Size(56, 24);
			this.buttonCancel.TabIndex = 7;
			this.buttonCancel.Text = "Cancel";
			this.buttonCancel.Click += new System.EventHandler(this.buttonCancel_Click);
			// 
			// DBLoginForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(290, 136);
			this.Controls.Add(this.buttonCancel);
			this.Controls.Add(this.buttonOK);
			this.Controls.Add(this.textPwd);
			this.Controls.Add(this.label3);
			this.Controls.Add(this.textUser);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.textDB);
			this.Controls.Add(this.label1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "DBLoginForm";
			this.Text = "DB Login";
			this.ResumeLayout(false);

		}
		#endregion

		private void buttonCancel_Click(object sender, System.EventArgs e)
		{
			DialogResult = DialogResult.Cancel;
			Close();
		}

		private void buttonOK_Click(object sender, System.EventArgs e)
		{
			DialogResult = DialogResult.OK;
			Close();
		}
	}
}
