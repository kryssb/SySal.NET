using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.OperaPublicationManager
{
	/// <summary>
	/// AddBricksForm is used to specify the bricks for which new publication jobs have to be created.
	/// </summary>
	/// <remarks>
	/// <para>The remote DB link to be involved in the operation is displayed in the related box.</para>
	/// <para>The IDs of the bricks to be involved must be entered as a comma-separated list in the related box. Further checks can be added:
	/// <list type="bullet">	
	/// <item><term>Not published yet</term><description>requires that the bricks haven't yet been published. Enabling this checks the <b><u>local</u></b> publication subsystem.</description></item>
	/// <item><term>Missing in destination</term><description>requires that the bricks are not in the destination: if the job type is <c>PUBLISH</c>/<c>UNPUBLISH</c>, <i>destination</i> means the remote DB link; if the job type is <c>COPY</c>/<c>DELETE</c>, <i>destination</i> means the local DB.</description></item>
	/// </list>
	/// The user clicks the <c>Select</c> button to have the form choose the bricks in the ID list and display them in the <c>Available bricks</c> table; notice that if a brick does not exist in the source DB, it will not appear in the <c>Available bricks</c> table.	
	/// </para>
	/// <para>The user highlights the bricks for which he/she wants to create jobs; no action will be taken for unselected bricks. The highlighted bricks are then moved to the list of <c>Selected bricks</c> on the right. Individual bricks or brick groups can be removed from that list by clicking on the <c>Remove</c> button, and the list can be cleared by clicking on the <c>Clear</c> button.</para>
	/// <para>New jobs are created for selected bricks if the user presses <c>OK</c>; no new jobs will be created if the user presses <c>Cancel</c>.</para>
	/// </remarks>
	public class AddBricksForm : System.Windows.Forms.Form
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
		private SySal.Controls.Button buttonSelect;
		private SySal.Controls.Button buttonPublish;
		private SySal.Controls.Button buttonClear;
		private SySal.Controls.Button buttonRemove;
		private SySal.Controls.Button buttonOKPublish;
		private SySal.Controls.CheckBox checkMissing;
		private SySal.Controls.CheckBox checkUnpublished;
		private SySal.Controls.GroupBox groupBox1;
		private SySal.Controls.GroupBox groupBox2;
		private SySal.Controls.Button buttonCancel;
		private SySal.Controls.StaticText staticText3;
		private SySal.Controls.StaticText staticUnpublished;
		private SySal.Controls.StaticText staticMissing;
		private SySal.Controls.StaticText staticDBLink;
		private SySal.Controls.StaticText staticHelp;
		private System.Windows.Forms.TextBox textIDList;
	
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
				case SySal.Executables.OperaPublicationManager.MainForm.JobType.Compare:		this.Text = "Add Brick 'COMPARE' Jobs"; staticDBLink.Visible = true; textDBLink.Visible = true; checkUnpublished.Visible = false; staticUnpublished.Visible = false; checkMissing.Visible = true; staticMissing.Visible = true; 
											staticHelp.LabelText = "Please select the bricks whose local data you want to compare with data stored on the remote DB.\r\nComparison involves both signature checking and row count comparison."; break;

				case SySal.Executables.OperaPublicationManager.MainForm.JobType.Publish:		this.Text = "Add Brick 'PUBLISH' Jobs"; staticDBLink.Visible = true; textDBLink.Visible = true; checkUnpublished.Visible = true; staticUnpublished.Visible = true; checkMissing.Visible = true; staticMissing.Visible = true; 
											staticHelp.LabelText = "Please select the bricks whose local data you want to upload to the remote DB."; break;

				case SySal.Executables.OperaPublicationManager.MainForm.JobType.Unpublish:		this.Text = "Add Brick 'UNPUBLISH' Jobs"; staticDBLink.Visible = true; textDBLink.Visible = true; checkUnpublished.Visible = true; staticUnpublished.Visible = true; checkMissing.Visible = true; staticMissing.Visible = true; 
											staticHelp.LabelText = "Please select the bricks whose local data you had published and you want now to remove\r\nfrom the remote DB."; break;

				case SySal.Executables.OperaPublicationManager.MainForm.JobType.CopyFromLink:	this.Text = "Add Brick 'COPY FROM LINK' Jobs"; staticDBLink.Visible = true; textDBLink.Visible = true; checkUnpublished.Visible = true; staticUnpublished.Visible = true; checkMissing.Visible = true; staticMissing.Visible = true; 
											staticHelp.LabelText = "Please select the bricks whose data you want to download from the remote DB."; break;

				case SySal.Executables.OperaPublicationManager.MainForm.JobType.DeleteLocal:	this.Text = "Add Brick 'DELETE LOCAL' Jobs"; staticDBLink.Visible = false; textDBLink.Visible = false; checkUnpublished.Visible = false; staticUnpublished.Visible = false; checkMissing.Visible = false; staticMissing.Visible = false;
											staticHelp.LabelText = "Please select the bricks whose data you want to delete from the local DB."; break;
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
		/// Creates a new AddBricksForm.
		/// </summary>
		public AddBricksForm()
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
			this.buttonSelect = new SySal.Controls.Button();
			this.buttonPublish = new SySal.Controls.Button();
			this.buttonClear = new SySal.Controls.Button();
			this.buttonRemove = new SySal.Controls.Button();
			this.buttonOKPublish = new SySal.Controls.Button();
			this.buttonCancel = new SySal.Controls.Button();
			this.staticUnpublished = new SySal.Controls.StaticText();
			this.staticMissing = new SySal.Controls.StaticText();
			this.checkMissing = new SySal.Controls.CheckBox();
			this.checkUnpublished = new SySal.Controls.CheckBox();
			this.staticDBLink = new SySal.Controls.StaticText();
			this.groupBox1 = new SySal.Controls.GroupBox();
			this.groupBox2 = new SySal.Controls.GroupBox();
			this.staticText3 = new SySal.Controls.StaticText();
			this.textIDList = new System.Windows.Forms.TextBox();
			this.staticHelp = new SySal.Controls.StaticText();
			((System.ComponentModel.ISupportInitialize)(this.gridLocal)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.gridPublish)).BeginInit();
			this.SuspendLayout();
			// 
			// textDBLink
			// 
			this.textDBLink.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(192)), ((System.Byte)(255)));
			this.textDBLink.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textDBLink.ForeColor = System.Drawing.Color.Navy;
			this.textDBLink.Location = new System.Drawing.Point(144, 80);
			this.textDBLink.Name = "textDBLink";
			this.textDBLink.ReadOnly = true;
			this.textDBLink.Size = new System.Drawing.Size(560, 20);
			this.textDBLink.TabIndex = 1;
			this.textDBLink.Text = "";
			// 
			// gridLocal
			// 
			this.gridLocal.AlternatingBackColor = System.Drawing.Color.White;
			this.gridLocal.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(255)), ((System.Byte)(255)));
			this.gridLocal.BackgroundColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(192)), ((System.Byte)(255)));
			this.gridLocal.CaptionBackColor = System.Drawing.Color.Navy;
			this.gridLocal.CaptionFont = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridLocal.CaptionForeColor = System.Drawing.Color.White;
			this.gridLocal.CaptionText = "Available bricks";
			this.gridLocal.CaptionVisible = false;
			this.gridLocal.DataMember = "";
			this.gridLocal.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridLocal.ForeColor = System.Drawing.Color.Navy;
			this.gridLocal.GridLineColor = System.Drawing.Color.FromArgb(((System.Byte)(0)), ((System.Byte)(0)), ((System.Byte)(64)));
			this.gridLocal.HeaderBackColor = System.Drawing.Color.FromArgb(((System.Byte)(128)), ((System.Byte)(128)), ((System.Byte)(255)));
			this.gridLocal.HeaderForeColor = System.Drawing.Color.Navy;
			this.gridLocal.Location = new System.Drawing.Point(32, 136);
			this.gridLocal.Name = "gridLocal";
			this.gridLocal.ReadOnly = true;
			this.gridLocal.SelectionBackColor = System.Drawing.Color.FromArgb(((System.Byte)(255)), ((System.Byte)(192)), ((System.Byte)(192)));
			this.gridLocal.SelectionForeColor = System.Drawing.Color.Navy;
			this.gridLocal.Size = new System.Drawing.Size(384, 216);
			this.gridLocal.TabIndex = 0;
			// 
			// gridPublish
			// 
			this.gridPublish.AlternatingBackColor = System.Drawing.Color.White;
			this.gridPublish.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(255)), ((System.Byte)(255)));
			this.gridPublish.BackgroundColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(192)), ((System.Byte)(255)));
			this.gridPublish.CaptionBackColor = System.Drawing.Color.Navy;
			this.gridPublish.CaptionFont = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridPublish.CaptionForeColor = System.Drawing.Color.White;
			this.gridPublish.CaptionText = "Bricks selected";
			this.gridPublish.CaptionVisible = false;
			this.gridPublish.DataMember = "";
			this.gridPublish.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridPublish.ForeColor = System.Drawing.Color.Navy;
			this.gridPublish.GridLineColor = System.Drawing.Color.FromArgb(((System.Byte)(0)), ((System.Byte)(0)), ((System.Byte)(64)));
			this.gridPublish.HeaderBackColor = System.Drawing.Color.FromArgb(((System.Byte)(128)), ((System.Byte)(128)), ((System.Byte)(255)));
			this.gridPublish.HeaderFont = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridPublish.HeaderForeColor = System.Drawing.Color.Navy;
			this.gridPublish.Location = new System.Drawing.Point(496, 136);
			this.gridPublish.Name = "gridPublish";
			this.gridPublish.PreferredColumnWidth = 120;
			this.gridPublish.ReadOnly = true;
			this.gridPublish.SelectionBackColor = System.Drawing.Color.FromArgb(((System.Byte)(255)), ((System.Byte)(192)), ((System.Byte)(192)));
			this.gridPublish.SelectionForeColor = System.Drawing.Color.Navy;
			this.gridPublish.Size = new System.Drawing.Size(200, 232);
			this.gridPublish.TabIndex = 5;
			// 
			// backgroundPanel1
			// 
			this.backgroundPanel1.BackColor = System.Drawing.Color.White;
			this.backgroundPanel1.Location = new System.Drawing.Point(0, 0);
			this.backgroundPanel1.Name = "backgroundPanel1";
			this.backgroundPanel1.Size = new System.Drawing.Size(720, 488);
			this.backgroundPanel1.TabIndex = 34;
			// 
			// buttonSelect
			// 
			this.buttonSelect.BackColor = System.Drawing.Color.White;
			this.buttonSelect.ButtonText = "Select";
			this.buttonSelect.Location = new System.Drawing.Point(32, 440);
			this.buttonSelect.Name = "buttonSelect";
			this.buttonSelect.Size = new System.Drawing.Size(80, 24);
			this.buttonSelect.TabIndex = 36;
			this.buttonSelect.Click += new System.EventHandler(this.buttonSelect_Click);
			this.buttonSelect.DoubleClick += new System.EventHandler(this.buttonSelect_Click);
			// 
			// buttonPublish
			// 
			this.buttonPublish.BackColor = System.Drawing.Color.White;
			this.buttonPublish.ButtonText = ">>";
			this.buttonPublish.Location = new System.Drawing.Point(440, 176);
			this.buttonPublish.Name = "buttonPublish";
			this.buttonPublish.Size = new System.Drawing.Size(32, 48);
			this.buttonPublish.TabIndex = 38;
			this.buttonPublish.Click += new System.EventHandler(this.buttonPublish_Click);
			this.buttonPublish.DoubleClick += new System.EventHandler(this.buttonPublish_Click);
			// 
			// buttonClear
			// 
			this.buttonClear.BackColor = System.Drawing.Color.White;
			this.buttonClear.ButtonText = "Clear";
			this.buttonClear.Location = new System.Drawing.Point(496, 376);
			this.buttonClear.Name = "buttonClear";
			this.buttonClear.Size = new System.Drawing.Size(80, 24);
			this.buttonClear.TabIndex = 39;
			this.buttonClear.Click += new System.EventHandler(this.buttonClear_Click);
			this.buttonClear.DoubleClick += new System.EventHandler(this.buttonClear_Click);
			// 
			// buttonRemove
			// 
			this.buttonRemove.BackColor = System.Drawing.Color.White;
			this.buttonRemove.ButtonText = "Remove";
			this.buttonRemove.Location = new System.Drawing.Point(616, 376);
			this.buttonRemove.Name = "buttonRemove";
			this.buttonRemove.Size = new System.Drawing.Size(80, 24);
			this.buttonRemove.TabIndex = 40;
			this.buttonRemove.Click += new System.EventHandler(this.buttonRemove_Click);
			this.buttonRemove.DoubleClick += new System.EventHandler(this.buttonRemove_Click);
			// 
			// buttonOKPublish
			// 
			this.buttonOKPublish.BackColor = System.Drawing.Color.White;
			this.buttonOKPublish.ButtonText = "OK";
			this.buttonOKPublish.Location = new System.Drawing.Point(624, 416);
			this.buttonOKPublish.Name = "buttonOKPublish";
			this.buttonOKPublish.Size = new System.Drawing.Size(80, 24);
			this.buttonOKPublish.TabIndex = 41;
			this.buttonOKPublish.Click += new System.EventHandler(this.buttonOKPublish_Click);
			this.buttonOKPublish.DoubleClick += new System.EventHandler(this.buttonOKPublish_Click);
			// 
			// buttonCancel
			// 
			this.buttonCancel.BackColor = System.Drawing.Color.White;
			this.buttonCancel.ButtonText = "Cancel";
			this.buttonCancel.Location = new System.Drawing.Point(624, 448);
			this.buttonCancel.Name = "buttonCancel";
			this.buttonCancel.Size = new System.Drawing.Size(80, 24);
			this.buttonCancel.TabIndex = 44;
			this.buttonCancel.Click += new System.EventHandler(this.buttonCancel_Click);
			this.buttonCancel.DoubleClick += new System.EventHandler(this.buttonCancel_Click);
			// 
			// staticUnpublished
			// 
			this.staticUnpublished.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticUnpublished.LabelText = "Not published yet";
			this.staticUnpublished.Location = new System.Drawing.Point(56, 392);
			this.staticUnpublished.Name = "staticUnpublished";
			this.staticUnpublished.Size = new System.Drawing.Size(120, 24);
			this.staticUnpublished.TabIndex = 49;
			this.staticUnpublished.TextAlign = System.Drawing.ContentAlignment.TopLeft;
			// 
			// staticMissing
			// 
			this.staticMissing.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticMissing.LabelText = "Missing in destination";
			this.staticMissing.Location = new System.Drawing.Point(56, 416);
			this.staticMissing.Name = "staticMissing";
			this.staticMissing.Size = new System.Drawing.Size(144, 24);
			this.staticMissing.TabIndex = 50;
			this.staticMissing.TextAlign = System.Drawing.ContentAlignment.TopLeft;
			// 
			// checkMissing
			// 
			this.checkMissing.BackColor = System.Drawing.Color.White;
			this.checkMissing.Checked = false;
			this.checkMissing.Location = new System.Drawing.Point(32, 416);
			this.checkMissing.Name = "checkMissing";
			this.checkMissing.Size = new System.Drawing.Size(16, 16);
			this.checkMissing.TabIndex = 53;
			// 
			// checkUnpublished
			// 
			this.checkUnpublished.BackColor = System.Drawing.Color.White;
			this.checkUnpublished.Checked = true;
			this.checkUnpublished.Location = new System.Drawing.Point(32, 392);
			this.checkUnpublished.Name = "checkUnpublished";
			this.checkUnpublished.Size = new System.Drawing.Size(16, 16);
			this.checkUnpublished.TabIndex = 54;
			// 
			// staticDBLink
			// 
			this.staticDBLink.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
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
			this.groupBox1.LabelText = "Available bricks";
			this.groupBox1.Location = new System.Drawing.Point(16, 104);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.OpenPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
			this.groupBox1.Size = new System.Drawing.Size(416, 368);
			this.groupBox1.TabIndex = 56;
			// 
			// groupBox2
			// 
			this.groupBox2.BackColor = System.Drawing.Color.White;
			this.groupBox2.ClosedPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
			this.groupBox2.IsOpen = true;
			this.groupBox2.IsStatic = true;
			this.groupBox2.LabelText = "Bricks selected";
			this.groupBox2.Location = new System.Drawing.Point(480, 104);
			this.groupBox2.Name = "groupBox2";
			this.groupBox2.OpenPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
			this.groupBox2.Size = new System.Drawing.Size(224, 304);
			this.groupBox2.TabIndex = 57;
			// 
			// staticText3
			// 
			this.staticText3.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText3.LabelText = "ID list (comma separated)";
			this.staticText3.Location = new System.Drawing.Point(32, 360);
			this.staticText3.Name = "staticText3";
			this.staticText3.Size = new System.Drawing.Size(168, 24);
			this.staticText3.TabIndex = 59;
			this.staticText3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// textIDList
			// 
			this.textIDList.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(255)), ((System.Byte)(255)));
			this.textIDList.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textIDList.ForeColor = System.Drawing.Color.Navy;
			this.textIDList.Location = new System.Drawing.Point(208, 360);
			this.textIDList.Name = "textIDList";
			this.textIDList.Size = new System.Drawing.Size(208, 20);
			this.textIDList.TabIndex = 58;
			this.textIDList.Text = "";
			// 
			// staticHelp
			// 
			this.staticHelp.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticHelp.LabelText = "Help Text";
			this.staticHelp.Location = new System.Drawing.Point(16, 16);
			this.staticHelp.Name = "staticHelp";
			this.staticHelp.Size = new System.Drawing.Size(688, 56);
			this.staticHelp.TabIndex = 60;
			this.staticHelp.TextAlign = System.Drawing.ContentAlignment.TopLeft;
			// 
			// AddBricksForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(722, 490);
			this.Controls.Add(this.staticHelp);
			this.Controls.Add(this.staticText3);
			this.Controls.Add(this.textIDList);
			this.Controls.Add(this.staticDBLink);
			this.Controls.Add(this.checkUnpublished);
			this.Controls.Add(this.checkMissing);
			this.Controls.Add(this.staticMissing);
			this.Controls.Add(this.staticUnpublished);
			this.Controls.Add(this.buttonCancel);
			this.Controls.Add(this.buttonOKPublish);
			this.Controls.Add(this.buttonRemove);
			this.Controls.Add(this.buttonClear);
			this.Controls.Add(this.buttonPublish);
			this.Controls.Add(this.buttonSelect);
			this.Controls.Add(this.gridPublish);
			this.Controls.Add(this.textDBLink);
			this.Controls.Add(this.gridLocal);
			this.Controls.Add(this.groupBox1);
			this.Controls.Add(this.groupBox2);
			this.Controls.Add(this.backgroundPanel1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "AddBricksForm";
			this.Text = "Add Brick Jobs";
			((System.ComponentModel.ISupportInitialize)(this.gridLocal)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.gridPublish)).EndInit();
			this.ResumeLayout(false);

		}
		#endregion

		private void buttonSelect_Click(object sender, System.EventArgs e)
		{
			string idstr = (string)textIDList.Text.Clone();
			long [] ids = null;
			string wherelist = null;
			try
			{
				string [] idss = idstr.Split(',');
				ids = new long[idss.Length];
				int i;
				wherelist = "";
				for (i = 0; i < ids.Length; i++)
				{
					ids[i] = Convert.ToInt64(idss[i].Trim());
					if (i > 0) wherelist += ",";
					wherelist += ids[i];
				}
			}
			catch (Exception x)
			{
				MessageBox.Show("Format error in ID List", "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}

			string selstr = null;
			switch (AddJobType)
			{
				case SySal.Executables.OperaPublicationManager.MainForm.JobType.Compare:
				{
					if (checkMissing.Checked)
						selstr = "SELECT ID FROM TB_EVENTBRICKS@" + textDBLink.Text + " WHERE ID IN (SELECT ID FROM TB_EVENTBRICKS WHERE ID IN (" + wherelist + "))";
					else 
						selstr = "SELECT ID FROM TB_EVENTBRICKS WHERE ID IN (" + wherelist + ")";
				}
					break;

				case SySal.Executables.OperaPublicationManager.MainForm.JobType.Publish:
				{
					if (checkMissing.Checked)
					{
						if (checkUnpublished.Checked) selstr = "SELECT ID FROM TB_EVENTBRICKS@" + textDBLink.Text + " WHERE ID IN (SELECT ID FROM TB_EVENTBRICKS WHERE ID IN (" + wherelist + ")) AND ID NOT IN (SELECT OBJID FROM PT_OBJECTS WHERE TYPE = 'BRICK')";
						else selstr = "SELECT ID FROM TB_EVENTBRICKS@" + textDBLink.Text + " WHERE ID IN (SELECT ID FROM TB_EVENTBRICKS WHERE ID IN (" + wherelist + "))";
					}
					else 
					{
						if (checkUnpublished.Checked) selstr = "SELECT ID FROM TB_EVENTBRICKS WHERE ID IN (" + wherelist + ") AND ID NOT IN (SELECT OBJID FROM PT_OBJECTS WHERE TYPE = 'BRICK')";
						else selstr = "SELECT ID FROM TB_EVENTBRICKS WHERE ID IN (" + wherelist + ")";
					}
				}
					break;
				
				case SySal.Executables.OperaPublicationManager.MainForm.JobType.Unpublish:
				{
					if (checkMissing.Checked)
					{
						if (checkUnpublished.Checked) selstr = "SELECT ID FROM TB_EVENTBRICKS@" + textDBLink.Text + " WHERE ID IN (SELECT ID FROM TB_EVENTBRICKS WHERE ID IN (" + wherelist + ")) AND ID NOT IN (SELECT OBJID FROM PT_OBJECTS WHERE TYPE = 'BRICK')";
						else selstr = "SELECT ID FROM TB_EVENTBRICKS@" + textDBLink.Text + " WHERE ID IN (SELECT ID FROM TB_EVENTBRICKS WHERE ID IN (" + wherelist + "))";
					}
					else 
					{
						if (checkUnpublished.Checked) selstr = "SELECT ID FROM TB_EVENTBRICKS WHERE ID IN (" + wherelist + ") AND ID NOT IN (SELECT OBJID FROM PT_OBJECTS WHERE TYPE = 'BRICK')";
						else selstr = "SELECT ID FROM TB_EVENTBRICKS WHERE ID IN (" + wherelist + ")";
					}
				}
					break;

				case SySal.Executables.OperaPublicationManager.MainForm.JobType.CopyFromLink:
				{
					if (checkMissing.Checked)
					{
						if (checkUnpublished.Checked) selstr = "SELECT ID FROM TB_EVENTBRICKS WHERE ID IN (SELECT ID FROM TB_EVENTBRICKS@" + textDBLink.Text + " WHERE ID IN (" + wherelist + ")) AND ID NOT IN (SELECT OBJID FROM PT_OBJECTS WHERE TYPE = 'BRICK')";
						else selstr = "SELECT ID FROM TB_EVENTBRICKS WHERE ID IN (SELECT ID FROM TB_EVENTBRICKS@" + textDBLink.Text + " WHERE ID IN (" + wherelist + "))";
					}
					else 
					{
						if (checkUnpublished.Checked) selstr = "SELECT ID FROM TB_EVENTBRICKS@" + textDBLink.Text + " WHERE ID IN (" + wherelist + ") AND ID NOT IN (SELECT OBJID FROM PT_OBJECTS WHERE TYPE = 'BRICK')";
						else selstr = "SELECT ID FROM TB_EVENTBRICKS@" + textDBLink.Text + " WHERE ID IN (" + wherelist + ")";
					}
				}
					break;

				case SySal.Executables.OperaPublicationManager.MainForm.JobType.DeleteLocal:
				{
					selstr = "SELECT ID FROM TB_EVENTBRICKS WHERE ID IN (" + wherelist + ")";
				}
					break;
			};
			Cursor oldc = this.Cursor;
			this.Cursor = Cursors.WaitCursor;
			System.Data.DataSet ds = new System.Data.DataSet();
			try
			{
				new SySal.OperaDb.OperaDbDataAdapter(selstr, Conn, null).Fill(ds);
				gridLocal.DataSource = ds.Tables[0];
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error in selection", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
			this.Cursor = oldc;
		}

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

		private void buttonOKPublish_Click(object sender, System.EventArgs e)
		{
			int i;
			Ids = new string[tablePublish.Rows.Count];
			for (i = 0; i < tablePublish.Rows.Count; i++)
				Ids[i] = gridPublish[i, 0].ToString();			
			DialogResult = DialogResult.OK;
		}

		/// <summary>
		/// The list of the brick Ids.
		/// </summary>
		internal string [] Ids = new string[0];

		private SySal.Executables.OperaPublicationManager.MainForm.JobType AddJobType;

		private void buttonCancel_Click(object sender, System.EventArgs e)
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
