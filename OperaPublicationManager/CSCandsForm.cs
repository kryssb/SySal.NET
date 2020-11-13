using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.OperaPublicationManager
{
	/// <summary>
	/// CSCandsForm is used to download CS candidates.
	/// </summary>
	/// <remarks>
	/// <para>The remote DB link to be involved in the operation is displayed in the related box.</para>
	/// <para>The IDs of the CS doublets for which the DB link contains predictions are loaded in the left list.</para>
	/// <para>The user highlights the bricks for which he/she wants to create jobs; no action will be taken for unselected bricks. The highlighted bricks are then moved to the list of <c>Selected bricks</c> on the right. Individual bricks or brick groups can be removed from that list by clicking on the <c>Remove</c> button, and the list can be cleared by clicking on the <c>Clear</c> button.</para>
	/// <para>Predictions are downloaded when the <c>Download</c> button is pressed. CS doublets are removed from the download list as the related candidates are downloaded; in case of errors, only the remaining doublets will be displayed.</para>
	/// </remarks>
	public class CSCandsForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.TextBox textDBLink;
		private System.Windows.Forms.DataGrid gridLocal;
		private System.Windows.Forms.DataGrid gridPublish;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		private SySal.OperaDb.OperaDbConnection Conn = null;

        private SySal.Controls.BackgroundPanel backgroundPanel1;
		private SySal.Controls.Button buttonPublish;
		private SySal.Controls.Button buttonClear;
		private SySal.Controls.Button buttonRemove;
        private SySal.Controls.Button buttonDownload;
		private SySal.Controls.GroupBox groupBox1;
		private SySal.Controls.GroupBox groupBox2;
        private SySal.Controls.Button buttonExit;
		private SySal.Controls.StaticText staticDBLink;
        private SySal.Controls.StaticText staticHelp;
	
		/// <summary>
		/// Shows the dialog.
		/// </summary>
		/// <param name="conn">the DB connection to be used.</param>
		/// <param name="dblink">the DB link involved in the jobs to be prepared.</param>
		/// <param name="jt">the type of job to be created.</param>
		/// <returns><c>DialogResult.OK</c> if the user presses <c>OK</c>, other codes otherwise.</returns>
		internal DialogResult ShowDialog(SySal.OperaDb.OperaDbConnection conn, string dblink, SySal.Executables.OperaPublicationManager.MainForm.JobType jt)
		{
			Conn = conn;
			textDBLink.Text = dblink;
			DialogResult res = DialogResult.Cancel;
			AddJobType = jt;
			switch (jt)
			{
				case SySal.Executables.OperaPublicationManager.MainForm.JobType.CSCandidates: staticDBLink.Visible = true; textDBLink.Visible = true; 
											staticHelp.LabelText = "Please select the CS doublets whose candidates you want to download.";
                                            Cursor oldc = Cursor;
                                            try
                                            {
                                                Cursor = Cursors.WaitCursor;
                                                System.Data.DataSet ds = new System.Data.DataSet();
                                                new SySal.OperaDb.OperaDbDataAdapter("SELECT DISTINCT ID_EVENTBRICK FROM VW_CS_CANDIDATES@" + dblink + " ORDER BY ID_EVENTBRICK", conn).Fill(ds);
                                                gridLocal.DataSource = ds.Tables[0];
                                                Cursor = oldc;
                                            }
                                            catch (Exception x)
                                            {
                                                Cursor = oldc;
                                                MessageBox.Show(x.Message, "Error downloading list of CS doublets", MessageBoxButtons.OK, MessageBoxIcon.Error);
                                                return DialogResult.Cancel;
                                            }
                    break;
			}
			try
			{
				res = ShowDialog();
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error executing \"Add bricks\" form", MessageBoxButtons.OK, MessageBoxIcon.Error);
				res = DialogResult.Cancel;
			}
			Conn = null;			
			return res;
		}

		/// <summary>
		/// Creates a new CSCandsForm.
		/// </summary>
		public CSCandsForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
			tablePublish.Columns.Clear();
			tablePublish.Columns.Add("ID", typeof(long));
			gridPublish.DataSource = tablePublish;
		}

		System.Data.DataTable tablePublish = new System.Data.DataTable();

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
            this.textDBLink = new System.Windows.Forms.TextBox();
            this.gridLocal = new System.Windows.Forms.DataGrid();
            this.gridPublish = new System.Windows.Forms.DataGrid();
            this.backgroundPanel1 = new SySal.Controls.BackgroundPanel();
            this.buttonPublish = new SySal.Controls.Button();
            this.buttonClear = new SySal.Controls.Button();
            this.buttonRemove = new SySal.Controls.Button();
            this.buttonDownload = new SySal.Controls.Button();
            this.buttonExit = new SySal.Controls.Button();
            this.staticDBLink = new SySal.Controls.StaticText();
            this.groupBox1 = new SySal.Controls.GroupBox();
            this.groupBox2 = new SySal.Controls.GroupBox();
            this.staticHelp = new SySal.Controls.StaticText();
            ((System.ComponentModel.ISupportInitialize)(this.gridLocal)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.gridPublish)).BeginInit();
            this.SuspendLayout();
            // 
            // textDBLink
            // 
            this.textDBLink.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(192)))), ((int)(((byte)(192)))), ((int)(((byte)(255)))));
            this.textDBLink.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textDBLink.ForeColor = System.Drawing.Color.Navy;
            this.textDBLink.Location = new System.Drawing.Point(144, 80);
            this.textDBLink.Name = "textDBLink";
            this.textDBLink.ReadOnly = true;
            this.textDBLink.Size = new System.Drawing.Size(560, 20);
            this.textDBLink.TabIndex = 1;
            // 
            // gridLocal
            // 
            this.gridLocal.AlternatingBackColor = System.Drawing.Color.White;
            this.gridLocal.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(192)))), ((int)(((byte)(255)))), ((int)(((byte)(255)))));
            this.gridLocal.BackgroundColor = System.Drawing.Color.FromArgb(((int)(((byte)(192)))), ((int)(((byte)(192)))), ((int)(((byte)(255)))));
            this.gridLocal.CaptionBackColor = System.Drawing.Color.Navy;
            this.gridLocal.CaptionFont = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.gridLocal.CaptionForeColor = System.Drawing.Color.White;
            this.gridLocal.CaptionText = "Available bricks";
            this.gridLocal.CaptionVisible = false;
            this.gridLocal.DataMember = "";
            this.gridLocal.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.gridLocal.ForeColor = System.Drawing.Color.Navy;
            this.gridLocal.GridLineColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(0)))), ((int)(((byte)(64)))));
            this.gridLocal.HeaderBackColor = System.Drawing.Color.FromArgb(((int)(((byte)(128)))), ((int)(((byte)(128)))), ((int)(((byte)(255)))));
            this.gridLocal.HeaderForeColor = System.Drawing.Color.Navy;
            this.gridLocal.Location = new System.Drawing.Point(32, 136);
            this.gridLocal.Name = "gridLocal";
            this.gridLocal.ReadOnly = true;
            this.gridLocal.SelectionBackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(192)))), ((int)(((byte)(192)))));
            this.gridLocal.SelectionForeColor = System.Drawing.Color.Navy;
            this.gridLocal.Size = new System.Drawing.Size(384, 246);
            this.gridLocal.TabIndex = 0;
            // 
            // gridPublish
            // 
            this.gridPublish.AlternatingBackColor = System.Drawing.Color.White;
            this.gridPublish.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(192)))), ((int)(((byte)(255)))), ((int)(((byte)(255)))));
            this.gridPublish.BackgroundColor = System.Drawing.Color.FromArgb(((int)(((byte)(192)))), ((int)(((byte)(192)))), ((int)(((byte)(255)))));
            this.gridPublish.CaptionBackColor = System.Drawing.Color.Navy;
            this.gridPublish.CaptionFont = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.gridPublish.CaptionForeColor = System.Drawing.Color.White;
            this.gridPublish.CaptionText = "Bricks selected";
            this.gridPublish.CaptionVisible = false;
            this.gridPublish.DataMember = "";
            this.gridPublish.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.gridPublish.ForeColor = System.Drawing.Color.Navy;
            this.gridPublish.GridLineColor = System.Drawing.Color.FromArgb(((int)(((byte)(0)))), ((int)(((byte)(0)))), ((int)(((byte)(64)))));
            this.gridPublish.HeaderBackColor = System.Drawing.Color.FromArgb(((int)(((byte)(128)))), ((int)(((byte)(128)))), ((int)(((byte)(255)))));
            this.gridPublish.HeaderFont = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.gridPublish.HeaderForeColor = System.Drawing.Color.Navy;
            this.gridPublish.Location = new System.Drawing.Point(496, 136);
            this.gridPublish.Name = "gridPublish";
            this.gridPublish.PreferredColumnWidth = 120;
            this.gridPublish.ReadOnly = true;
            this.gridPublish.SelectionBackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(192)))), ((int)(((byte)(192)))));
            this.gridPublish.SelectionForeColor = System.Drawing.Color.Navy;
            this.gridPublish.Size = new System.Drawing.Size(200, 216);
            this.gridPublish.TabIndex = 5;
            // 
            // backgroundPanel1
            // 
            this.backgroundPanel1.BackColor = System.Drawing.Color.White;
            this.backgroundPanel1.Location = new System.Drawing.Point(0, 0);
            this.backgroundPanel1.Name = "backgroundPanel1";
            this.backgroundPanel1.Size = new System.Drawing.Size(720, 441);
            this.backgroundPanel1.TabIndex = 34;
            // 
            // buttonPublish
            // 
            this.buttonPublish.BackColor = System.Drawing.Color.White;
            this.buttonPublish.ButtonText = ">>";
            this.buttonPublish.Location = new System.Drawing.Point(440, 176);
            this.buttonPublish.Name = "buttonPublish";
            this.buttonPublish.Size = new System.Drawing.Size(32, 48);
            this.buttonPublish.TabIndex = 38;
            this.buttonPublish.DoubleClick += new System.EventHandler(this.buttonPublish_Click);
            this.buttonPublish.Click += new System.EventHandler(this.buttonPublish_Click);
            // 
            // buttonClear
            // 
            this.buttonClear.BackColor = System.Drawing.Color.White;
            this.buttonClear.ButtonText = "Clear";
            this.buttonClear.Location = new System.Drawing.Point(496, 358);
            this.buttonClear.Name = "buttonClear";
            this.buttonClear.Size = new System.Drawing.Size(80, 24);
            this.buttonClear.TabIndex = 39;
            this.buttonClear.DoubleClick += new System.EventHandler(this.buttonClear_Click);
            this.buttonClear.Click += new System.EventHandler(this.buttonClear_Click);
            // 
            // buttonRemove
            // 
            this.buttonRemove.BackColor = System.Drawing.Color.White;
            this.buttonRemove.ButtonText = "Remove";
            this.buttonRemove.Location = new System.Drawing.Point(616, 358);
            this.buttonRemove.Name = "buttonRemove";
            this.buttonRemove.Size = new System.Drawing.Size(80, 24);
            this.buttonRemove.TabIndex = 40;
            this.buttonRemove.DoubleClick += new System.EventHandler(this.buttonRemove_Click);
            this.buttonRemove.Click += new System.EventHandler(this.buttonRemove_Click);
            // 
            // buttonDownload
            // 
            this.buttonDownload.BackColor = System.Drawing.Color.White;
            this.buttonDownload.ButtonText = "Download";
            this.buttonDownload.Location = new System.Drawing.Point(624, 400);
            this.buttonDownload.Name = "buttonDownload";
            this.buttonDownload.Size = new System.Drawing.Size(80, 24);
            this.buttonDownload.TabIndex = 41;
            this.buttonDownload.DoubleClick += new System.EventHandler(this.buttonDownload_Click);
            this.buttonDownload.Click += new System.EventHandler(this.buttonDownload_Click);
            // 
            // buttonExit
            // 
            this.buttonExit.BackColor = System.Drawing.Color.White;
            this.buttonExit.ButtonText = "Exit";
            this.buttonExit.Location = new System.Drawing.Point(16, 400);
            this.buttonExit.Name = "buttonExit";
            this.buttonExit.Size = new System.Drawing.Size(80, 24);
            this.buttonExit.TabIndex = 44;
            this.buttonExit.DoubleClick += new System.EventHandler(this.buttonExit_Click);
            this.buttonExit.Click += new System.EventHandler(this.buttonExit_Click);
            // 
            // staticDBLink
            // 
            this.staticDBLink.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(126)))), ((int)(((byte)(181)))), ((int)(((byte)(232)))));
            this.staticDBLink.LabelText = "Remote DB Link";
            this.staticDBLink.Location = new System.Drawing.Point(16, 80);
            this.staticDBLink.Name = "staticDBLink";
            this.staticDBLink.Size = new System.Drawing.Size(128, 24);
            this.staticDBLink.TabIndex = 55;
            this.staticDBLink.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // groupBox1
            // 
            this.groupBox1.BackColor = System.Drawing.Color.White;
            this.groupBox1.ClosedPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
            this.groupBox1.IsOpen = true;
            this.groupBox1.IsStatic = true;
            this.groupBox1.LabelText = "Available CS doublets";
            this.groupBox1.Location = new System.Drawing.Point(16, 104);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.OpenPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
            this.groupBox1.Size = new System.Drawing.Size(416, 290);
            this.groupBox1.TabIndex = 56;
            // 
            // groupBox2
            // 
            this.groupBox2.BackColor = System.Drawing.Color.White;
            this.groupBox2.ClosedPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
            this.groupBox2.IsOpen = true;
            this.groupBox2.IsStatic = true;
            this.groupBox2.LabelText = "CS doublets selected";
            this.groupBox2.Location = new System.Drawing.Point(480, 104);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.OpenPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
            this.groupBox2.Size = new System.Drawing.Size(224, 290);
            this.groupBox2.TabIndex = 57;
            // 
            // staticHelp
            // 
            this.staticHelp.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(126)))), ((int)(((byte)(181)))), ((int)(((byte)(232)))));
            this.staticHelp.LabelText = "Help Text";
            this.staticHelp.Location = new System.Drawing.Point(16, 16);
            this.staticHelp.Name = "staticHelp";
            this.staticHelp.Size = new System.Drawing.Size(688, 56);
            this.staticHelp.TabIndex = 60;
            this.staticHelp.TextAlign = System.Drawing.ContentAlignment.TopLeft;
            // 
            // CSCandsForm
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(722, 443);
            this.Controls.Add(this.staticHelp);
            this.Controls.Add(this.staticDBLink);
            this.Controls.Add(this.buttonExit);
            this.Controls.Add(this.buttonDownload);
            this.Controls.Add(this.buttonRemove);
            this.Controls.Add(this.buttonClear);
            this.Controls.Add(this.buttonPublish);
            this.Controls.Add(this.gridPublish);
            this.Controls.Add(this.textDBLink);
            this.Controls.Add(this.gridLocal);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.backgroundPanel1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.Name = "CSCandsForm";
            this.Text = "Download CS Candidates";
            ((System.ComponentModel.ISupportInitialize)(this.gridLocal)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.gridPublish)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

		}
		#endregion

		private void buttonPublish_Click(object sender, System.EventArgs e)
		{
			int i, j;
			int len = ((System.Data.DataTable)gridLocal.DataSource).Rows.Count;
			for (i = 0; i < len; i++)
				if (gridLocal.IsSelected(i))
				{
					long x = SySal.OperaDb.Convert.ToInt64(gridLocal[i, 0].ToString());
					long y = -1;
					for (j = 0; j < tablePublish.Rows.Count; j++)
						if ((y = SySal.OperaDb.Convert.ToInt64(tablePublish.Rows[j][0])) >= x) 
							break;
					if (j < tablePublish.Rows.Count && y == x) continue;
					System.Data.DataRow dr = tablePublish.NewRow();
					dr[0] = x;
					tablePublish.Rows.InsertAt(dr, j);					
				} 
			System.Data.DataTable newPublish = tablePublish.Copy();
			gridPublish.DataSource = newPublish;
			tablePublish = newPublish;
		}

		private void buttonRemove_Click(object sender, System.EventArgs e)
		{
			int i, j;
			int len = tablePublish.Rows.Count;
			System.Data.DataTable newPublish = tablePublish.Copy();
			for (i = j = 0; i < len; i++, j++)
				if (gridPublish.IsSelected(i))
				{
					newPublish.Rows.RemoveAt(j--);
				} 		
			gridPublish.DataSource = newPublish;
			tablePublish = newPublish;
		}

		private void buttonDownload_Click(object sender, System.EventArgs e)
		{
            Cursor oldc = Cursor;
            try
            {
                Cursor = Cursors.WaitCursor;
                SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("CALL PP_DOWNLOAD_PREDICTIONS(:s, '" + this.textDBLink.Text + "')", Conn);
                cmd.Parameters.Add("s", SySal.OperaDb.OperaDbType.String, System.Data.ParameterDirection.Input);
                int i;
                while (tablePublish.Rows.Count > 0)
                {
                    cmd.Parameters[0].Value = tablePublish.Rows[0][0].ToString();
                    cmd.ExecuteNonQuery();
                    tablePublish.Rows.RemoveAt(0);
                }
                Cursor = oldc;
                MessageBox.Show("All predictions downloaded", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception x)
            {
                Cursor = oldc;
                MessageBox.Show(x.Message, "Error downloading prediction", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
			DialogResult = DialogResult.OK;
		}

		/// <summary>
		/// The list of the brick Ids.
		/// </summary>
		internal string [] Ids = new string[0];

		private SySal.Executables.OperaPublicationManager.MainForm.JobType AddJobType;

		private void buttonExit_Click(object sender, System.EventArgs e)
		{
			DialogResult = DialogResult.Cancel;
		}

		private void buttonClear_Click(object sender, System.EventArgs e)
		{
			tablePublish.Rows.Clear();
			gridPublish.DataSource = tablePublish;
        }

	}
}
