using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.OperaPublicationManager
{
	/// <summary>
	/// Allows the user to enter data to connect to a DB.
	/// </summary>
	/// <remarks>
	/// <list type="bullet">
	/// <item><term>DB Server</term><description>the name of the DB to which a link has to be established.</description></item>
	/// <item><term>Username</term><description>the name of the user under whose identity the DB link should be established.</description></item>
	/// <item><term>Password</term><description>password of the DB user name.</description></item>	
	/// <item><term>DB Link</term><description>the name to be assigned to the DB link.</description></item>
	/// </list>
	/// </remarks>
	public class DBLinkForm : System.Windows.Forms.Form
	{
		internal System.Windows.Forms.TextBox textDBServer;
		internal System.Windows.Forms.TextBox textUsername;
		internal System.Windows.Forms.TextBox textPassword;
		internal System.Windows.Forms.TextBox textLinkName;
		private SySal.Controls.BackgroundPanel backgroundPanel1;
		private SySal.Controls.StaticText staticText1;
		private SySal.Controls.StaticText staticText2;
		private SySal.Controls.StaticText staticText3;
		private SySal.Controls.StaticText staticText4;
		private SySal.Controls.Button buttonOK;
		private SySal.Controls.Button buttonCancel;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		/// <summary>
		/// Creates a new DBLinkForm.
		/// </summary>
		public DBLinkForm()
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
			this.textDBServer = new System.Windows.Forms.TextBox();
			this.textUsername = new System.Windows.Forms.TextBox();
			this.textPassword = new System.Windows.Forms.TextBox();
			this.textLinkName = new System.Windows.Forms.TextBox();
			this.backgroundPanel1 = new SySal.Controls.BackgroundPanel();
			this.staticText1 = new SySal.Controls.StaticText();
			this.staticText2 = new SySal.Controls.StaticText();
			this.staticText3 = new SySal.Controls.StaticText();
			this.staticText4 = new SySal.Controls.StaticText();
			this.buttonOK = new SySal.Controls.Button();
			this.buttonCancel = new SySal.Controls.Button();
			this.SuspendLayout();
			// 
			// textDBServer
			// 
			this.textDBServer.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(255)), ((System.Byte)(255)));
			this.textDBServer.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textDBServer.ForeColor = System.Drawing.Color.Navy;
			this.textDBServer.Location = new System.Drawing.Point(96, 16);
			this.textDBServer.Name = "textDBServer";
			this.textDBServer.Size = new System.Drawing.Size(176, 20);
			this.textDBServer.TabIndex = 2;
			this.textDBServer.Text = "";
			// 
			// textUsername
			// 
			this.textUsername.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(255)), ((System.Byte)(255)));
			this.textUsername.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textUsername.ForeColor = System.Drawing.Color.Navy;
			this.textUsername.Location = new System.Drawing.Point(96, 40);
			this.textUsername.Name = "textUsername";
			this.textUsername.Size = new System.Drawing.Size(176, 20);
			this.textUsername.TabIndex = 4;
			this.textUsername.Text = "";
			// 
			// textPassword
			// 
			this.textPassword.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(255)), ((System.Byte)(255)));
			this.textPassword.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textPassword.ForeColor = System.Drawing.Color.Navy;
			this.textPassword.Location = new System.Drawing.Point(96, 64);
			this.textPassword.Name = "textPassword";
			this.textPassword.PasswordChar = '*';
			this.textPassword.Size = new System.Drawing.Size(176, 20);
			this.textPassword.TabIndex = 6;
			this.textPassword.Text = "";
			// 
			// textLinkName
			// 
			this.textLinkName.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(255)), ((System.Byte)(255)));
			this.textLinkName.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textLinkName.ForeColor = System.Drawing.Color.Navy;
			this.textLinkName.Location = new System.Drawing.Point(96, 88);
			this.textLinkName.Name = "textLinkName";
			this.textLinkName.Size = new System.Drawing.Size(176, 20);
			this.textLinkName.TabIndex = 8;
			this.textLinkName.Text = "";
			// 
			// backgroundPanel1
			// 
			this.backgroundPanel1.BackColor = System.Drawing.Color.White;
			this.backgroundPanel1.Location = new System.Drawing.Point(0, 0);
			this.backgroundPanel1.Name = "backgroundPanel1";
			this.backgroundPanel1.Size = new System.Drawing.Size(296, 168);
			this.backgroundPanel1.TabIndex = 0;
			// 
			// staticText1
			// 
			this.staticText1.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText1.LabelText = "DB Server";
			this.staticText1.Location = new System.Drawing.Point(16, 16);
			this.staticText1.Name = "staticText1";
			this.staticText1.Size = new System.Drawing.Size(72, 24);
			this.staticText1.TabIndex = 1;
			this.staticText1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// staticText2
			// 
			this.staticText2.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText2.LabelText = "Username";
			this.staticText2.Location = new System.Drawing.Point(16, 40);
			this.staticText2.Name = "staticText2";
			this.staticText2.Size = new System.Drawing.Size(72, 24);
			this.staticText2.TabIndex = 3;
			this.staticText2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// staticText3
			// 
			this.staticText3.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText3.LabelText = "Password";
			this.staticText3.Location = new System.Drawing.Point(16, 64);
			this.staticText3.Name = "staticText3";
			this.staticText3.Size = new System.Drawing.Size(72, 24);
			this.staticText3.TabIndex = 5;
			this.staticText3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// staticText4
			// 
			this.staticText4.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText4.LabelText = "Link name";
			this.staticText4.Location = new System.Drawing.Point(16, 88);
			this.staticText4.Name = "staticText4";
			this.staticText4.Size = new System.Drawing.Size(72, 24);
			this.staticText4.TabIndex = 7;
			this.staticText4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// buttonOK
			// 
			this.buttonOK.BackColor = System.Drawing.Color.White;
			this.buttonOK.ButtonText = "OK";
			this.buttonOK.Location = new System.Drawing.Point(216, 120);
			this.buttonOK.Name = "buttonOK";
			this.buttonOK.Size = new System.Drawing.Size(56, 24);
			this.buttonOK.TabIndex = 10;
			this.buttonOK.Click += new System.EventHandler(this.buttonOK_Click);
			this.buttonOK.DoubleClick += new System.EventHandler(this.buttonOK_Click);
			// 
			// buttonCancel
			// 
			this.buttonCancel.BackColor = System.Drawing.Color.White;
			this.buttonCancel.ButtonText = "Cancel";
			this.buttonCancel.Location = new System.Drawing.Point(16, 120);
			this.buttonCancel.Name = "buttonCancel";
			this.buttonCancel.Size = new System.Drawing.Size(56, 24);
			this.buttonCancel.TabIndex = 9;
			this.buttonCancel.Click += new System.EventHandler(this.buttonCancel_Click);
			this.buttonCancel.DoubleClick += new System.EventHandler(this.buttonCancel_Click);
			// 
			// DBLinkForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(298, 168);
			this.Controls.Add(this.buttonCancel);
			this.Controls.Add(this.buttonOK);
			this.Controls.Add(this.staticText4);
			this.Controls.Add(this.staticText3);
			this.Controls.Add(this.staticText2);
			this.Controls.Add(this.staticText1);
			this.Controls.Add(this.textLinkName);
			this.Controls.Add(this.textPassword);
			this.Controls.Add(this.textUsername);
			this.Controls.Add(this.textDBServer);
			this.Controls.Add(this.backgroundPanel1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "DBLinkForm";
			this.ShowInTaskbar = false;
			this.Text = "Login Information";
			this.ResumeLayout(false);

		}
		#endregion

		private void buttonOK_Click(object sender, System.EventArgs e)
		{
			DialogResult = DialogResult.OK;
			Close();
		}

		private void buttonCancel_Click(object sender, System.EventArgs e)
		{
			DialogResult = DialogResult.Cancel;
			Close();		
		}
	}
}
