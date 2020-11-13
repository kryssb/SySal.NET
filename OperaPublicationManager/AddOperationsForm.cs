using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.OperaPublicationManager
{
	/// <summary>
	/// AddOperationsForm is used to specify the process operations for which new publication jobs have to be created.
	/// </summary>
	/// <remarks>
	/// <para>The remote DB link to be involved in the operation is displayed in the related box.</para>
	/// <para>The IDs of the operations to be involved must be entered as a comma-separated list or as a single-column-returning SQL statement. Examples:
	/// <list type="bullet">
	/// <item><term>ID list</term><description><c>435,882,256,883</c></description></item>
	/// <item><term>SQL</term><description><c>SELECT DISTINCT ID_SCANBACK_PROCOPID FROM TB_B_SCANBACK_CHECKRESULTS WHERE ID_EVENTBRICK = 10081</c></description></item>
	/// </list> 
	/// Further checks can be added:
	/// <list type="bullet">	
	/// <item><term>Not published yet</term><description>requires that the process operations haven't yet been published. Enabling this checks the <b><u>local</u></b> publication subsystem.</description></item>
	/// <item><term>Missing in destination</term><description>requires that the process operations are not in the destination: if the job type is <c>PUBLISH</c>/<c>UNPUBLISH</c>, <i>destination</i> means the remote DB link; if the job type is <c>COPY</c>/<c>DELETE</c>, <i>destination</i> means the local DB.</description></item>
	/// </list>
	/// The user clicks the <c>Select</c> button to have the form choose the process operations in the ID list or in the SQL selection and display them in the <c>Available operations</c> table; notice that if a process operation does not exist in the source DB, it will not appear in the <c>Available operations</c> table.
	/// </para>
	/// <para>Process operation jobs use to fail when dependent process operations are not present in the destination. In order to avoid skipping dependencies, the user can press the <c>Dependencies</c> button to get a report of process operations that must be involved in the process operation list. Since this can be a long task, a progress bar is on the right of this button. The report is displayed in a <see cref="SySal.Executables.OperaPublicationManager.DependencyForm"/>.</para>
	/// <para>The user highlights the process operations for which he/she wants to create jobs; no action will be taken for unselected operations. The highlighted bricks are then moved to the list of <c>Selected bricks</c> on the right. Individual operations or operation groups can be removed from that list by clicking on the <c>Remove</c> button, and the list can be cleared by clicking on the <c>Clear</c> button.</para>
	/// <para>New jobs are created for selected operations if the user presses <c>OK</c>; no new jobs will be created if the user presses <c>Cancel</c>.</para>
	/// </remarks>
	public class AddOperationsForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.TextBox textDBLink;
		private System.Windows.Forms.DataGrid gridLocal;
		private System.Windows.Forms.DataGrid gridPublish;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		private SySal.OperaDb.OperaDbConnection Conn = null;

		bool ShouldStop = false;
		private System.Windows.Forms.TextBox textSQL;
		private SySal.Controls.BackgroundPanel backgroundPanel1;
		private SySal.Controls.Button buttonPublish;
		private SySal.Controls.Button buttonOKPublish;
		private SySal.Controls.Button buttonRemove;
		private SySal.Controls.Button buttonClear;
		private SySal.Controls.StaticText staticText3;
		private SySal.Controls.CheckBox checkUnpublished;
		private SySal.Controls.CheckBox checkMissing;
		private SySal.Controls.Button buttonDependencies;
		private SySal.Controls.Button buttonCancel;
		private SySal.Controls.Button buttonSelect;
		private SySal.Controls.GroupBox groupBox1;
		private SySal.Controls.GroupBox groupBox2;
		private SySal.Controls.ProgressBar progressBar1;
		private SySal.Controls.StaticText staticHelp;
		private SySal.Controls.StaticText staticDBLink;
		private SySal.Controls.StaticText staticMissing;
		private SySal.Controls.StaticText staticUnpublished;

		System.Threading.Thread ExecThread = null;

		/// <summary>
		/// Shows the dialog.
		/// </summary>
		/// <param name="conn">the DB connection to be used.</param>
		/// <param name="dblink">the DB link involved in the jobs to be prepared.</param>		
		/// <returns><c>DialogResult.OK</c> if the user presses <c>OK</c>, other codes otherwise.</returns>
		public DialogResult ShowDialog(SySal.OperaDb.OperaDbConnection conn, string dblink)
		{
			Conn = conn;
			textDBLink.Text = dblink;
			DialogResult res = DialogResult.Cancel;
			try
			{
				res = ShowDialog();
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error executing \"Add operations\" form", MessageBoxButtons.OK, MessageBoxIcon.Error);
				res = DialogResult.Cancel;
			}
			Conn = null;			
			return res;
		}

		/// <summary>
		/// Creates a new AddOperationsForm.
		/// </summary>
		public AddOperationsForm()
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
			this.textSQL = new System.Windows.Forms.TextBox();
			this.backgroundPanel1 = new SySal.Controls.BackgroundPanel();
			this.staticDBLink = new SySal.Controls.StaticText();
			this.buttonPublish = new SySal.Controls.Button();
			this.buttonCancel = new SySal.Controls.Button();
			this.buttonOKPublish = new SySal.Controls.Button();
			this.buttonRemove = new SySal.Controls.Button();
			this.buttonClear = new SySal.Controls.Button();
			this.staticText3 = new SySal.Controls.StaticText();
			this.checkUnpublished = new SySal.Controls.CheckBox();
			this.checkMissing = new SySal.Controls.CheckBox();
			this.staticMissing = new SySal.Controls.StaticText();
			this.staticUnpublished = new SySal.Controls.StaticText();
			this.buttonSelect = new SySal.Controls.Button();
			this.buttonDependencies = new SySal.Controls.Button();
			this.groupBox1 = new SySal.Controls.GroupBox();
			this.groupBox2 = new SySal.Controls.GroupBox();
			this.progressBar1 = new SySal.Controls.ProgressBar();
			this.staticHelp = new SySal.Controls.StaticText();
			((System.ComponentModel.ISupportInitialize)(this.gridLocal)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.gridPublish)).BeginInit();
			this.SuspendLayout();
			// 
			// textDBLink
			// 
			this.textDBLink.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(192)), ((System.Byte)(255)));
			this.textDBLink.Location = new System.Drawing.Point(144, 56);
			this.textDBLink.Name = "textDBLink";
			this.textDBLink.ReadOnly = true;
			this.textDBLink.Size = new System.Drawing.Size(720, 20);
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
			this.gridLocal.CaptionText = "Available operations";
			this.gridLocal.CaptionVisible = false;
			this.gridLocal.DataMember = "";
			this.gridLocal.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridLocal.ForeColor = System.Drawing.Color.Navy;
			this.gridLocal.GridLineColor = System.Drawing.Color.FromArgb(((System.Byte)(0)), ((System.Byte)(0)), ((System.Byte)(64)));
			this.gridLocal.HeaderBackColor = System.Drawing.Color.FromArgb(((System.Byte)(128)), ((System.Byte)(128)), ((System.Byte)(255)));
			this.gridLocal.HeaderForeColor = System.Drawing.Color.Navy;
			this.gridLocal.Location = new System.Drawing.Point(32, 120);
			this.gridLocal.Name = "gridLocal";
			this.gridLocal.PreferredColumnWidth = 120;
			this.gridLocal.ReadOnly = true;
			this.gridLocal.SelectionBackColor = System.Drawing.Color.FromArgb(((System.Byte)(255)), ((System.Byte)(192)), ((System.Byte)(192)));
			this.gridLocal.SelectionForeColor = System.Drawing.Color.Navy;
			this.gridLocal.Size = new System.Drawing.Size(536, 216);
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
			this.gridPublish.CaptionText = "Operations selected";
			this.gridPublish.CaptionVisible = false;
			this.gridPublish.DataMember = "";
			this.gridPublish.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.gridPublish.ForeColor = System.Drawing.Color.Navy;
			this.gridPublish.GridLineColor = System.Drawing.Color.FromArgb(((System.Byte)(0)), ((System.Byte)(0)), ((System.Byte)(64)));
			this.gridPublish.HeaderBackColor = System.Drawing.Color.FromArgb(((System.Byte)(128)), ((System.Byte)(128)), ((System.Byte)(255)));
			this.gridPublish.HeaderForeColor = System.Drawing.Color.Navy;
			this.gridPublish.Location = new System.Drawing.Point(648, 120);
			this.gridPublish.Name = "gridPublish";
			this.gridPublish.PreferredColumnWidth = 120;
			this.gridPublish.ReadOnly = true;
			this.gridPublish.SelectionBackColor = System.Drawing.Color.FromArgb(((System.Byte)(255)), ((System.Byte)(192)), ((System.Byte)(192)));
			this.gridPublish.SelectionForeColor = System.Drawing.Color.Navy;
			this.gridPublish.Size = new System.Drawing.Size(200, 320);
			this.gridPublish.TabIndex = 5;
			// 
			// textSQL
			// 
			this.textSQL.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textSQL.ForeColor = System.Drawing.Color.Navy;
			this.textSQL.Location = new System.Drawing.Point(32, 376);
			this.textSQL.Multiline = true;
			this.textSQL.Name = "textSQL";
			this.textSQL.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
			this.textSQL.Size = new System.Drawing.Size(536, 80);
			this.textSQL.TabIndex = 31;
			this.textSQL.Text = "SELECT ID FROM TB_PROC_OPERATIONS WHERE ...";
			// 
			// backgroundPanel1
			// 
			this.backgroundPanel1.BackColor = System.Drawing.Color.White;
			this.backgroundPanel1.Location = new System.Drawing.Point(0, 0);
			this.backgroundPanel1.Name = "backgroundPanel1";
			this.backgroundPanel1.Size = new System.Drawing.Size(880, 560);
			this.backgroundPanel1.TabIndex = 35;
			// 
			// staticDBLink
			// 
			this.staticDBLink.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticDBLink.LabelText = "Remote DB Link";
			this.staticDBLink.Location = new System.Drawing.Point(8, 56);
			this.staticDBLink.Name = "staticDBLink";
			this.staticDBLink.Size = new System.Drawing.Size(128, 24);
			this.staticDBLink.TabIndex = 56;
			this.staticDBLink.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// buttonPublish
			// 
			this.buttonPublish.BackColor = System.Drawing.Color.White;
			this.buttonPublish.ButtonText = ">>";
			this.buttonPublish.Location = new System.Drawing.Point(592, 160);
			this.buttonPublish.Name = "buttonPublish";
			this.buttonPublish.Size = new System.Drawing.Size(32, 48);
			this.buttonPublish.TabIndex = 57;
			this.buttonPublish.Click += new System.EventHandler(this.buttonPublish_Click);
			this.buttonPublish.DoubleClick += new System.EventHandler(this.buttonPublish_Click);
			// 
			// buttonCancel
			// 
			this.buttonCancel.BackColor = System.Drawing.Color.White;
			this.buttonCancel.ButtonText = "Cancel";
			this.buttonCancel.Location = new System.Drawing.Point(784, 520);
			this.buttonCancel.Name = "buttonCancel";
			this.buttonCancel.Size = new System.Drawing.Size(80, 24);
			this.buttonCancel.TabIndex = 63;
			this.buttonCancel.Click += new System.EventHandler(this.buttonCancel_Click);
			this.buttonCancel.DoubleClick += new System.EventHandler(this.buttonCancel_Click);
			// 
			// buttonOKPublish
			// 
			this.buttonOKPublish.BackColor = System.Drawing.Color.White;
			this.buttonOKPublish.ButtonText = "OK";
			this.buttonOKPublish.Location = new System.Drawing.Point(784, 488);
			this.buttonOKPublish.Name = "buttonOKPublish";
			this.buttonOKPublish.Size = new System.Drawing.Size(80, 24);
			this.buttonOKPublish.TabIndex = 60;
			this.buttonOKPublish.Click += new System.EventHandler(this.buttonOKPublish_Click);
			this.buttonOKPublish.DoubleClick += new System.EventHandler(this.buttonOKPublish_Click);
			// 
			// buttonRemove
			// 
			this.buttonRemove.BackColor = System.Drawing.Color.White;
			this.buttonRemove.ButtonText = "Remove";
			this.buttonRemove.Location = new System.Drawing.Point(768, 448);
			this.buttonRemove.Name = "buttonRemove";
			this.buttonRemove.Size = new System.Drawing.Size(80, 24);
			this.buttonRemove.TabIndex = 59;
			this.buttonRemove.Click += new System.EventHandler(this.buttonRemove_Click);
			this.buttonRemove.DoubleClick += new System.EventHandler(this.buttonRemove_Click);
			// 
			// buttonClear
			// 
			this.buttonClear.BackColor = System.Drawing.Color.White;
			this.buttonClear.ButtonText = "Clear";
			this.buttonClear.Location = new System.Drawing.Point(648, 448);
			this.buttonClear.Name = "buttonClear";
			this.buttonClear.Size = new System.Drawing.Size(80, 24);
			this.buttonClear.TabIndex = 58;
			this.buttonClear.Click += new System.EventHandler(this.buttonClear_Click);
			this.buttonClear.DoubleClick += new System.EventHandler(this.buttonClear_Click);
			// 
			// staticText3
			// 
			this.staticText3.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText3.LabelText = "ID list (comma separated) or SQL selection (must return a single column)";
			this.staticText3.Location = new System.Drawing.Point(32, 352);
			this.staticText3.Name = "staticText3";
			this.staticText3.Size = new System.Drawing.Size(496, 24);
			this.staticText3.TabIndex = 69;
			this.staticText3.TextAlign = System.Drawing.ContentAlignment.TopLeft;
			// 
			// checkUnpublished
			// 
			this.checkUnpublished.BackColor = System.Drawing.Color.White;
			this.checkUnpublished.Checked = true;
			this.checkUnpublished.Location = new System.Drawing.Point(32, 464);
			this.checkUnpublished.Name = "checkUnpublished";
			this.checkUnpublished.Size = new System.Drawing.Size(16, 16);
			this.checkUnpublished.TabIndex = 77;
			// 
			// checkMissing
			// 
			this.checkMissing.BackColor = System.Drawing.Color.White;
			this.checkMissing.Checked = false;
			this.checkMissing.Location = new System.Drawing.Point(32, 488);
			this.checkMissing.Name = "checkMissing";
			this.checkMissing.Size = new System.Drawing.Size(16, 16);
			this.checkMissing.TabIndex = 76;
			// 
			// staticMissing
			// 
			this.staticMissing.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticMissing.LabelText = "Missing in destination";
			this.staticMissing.Location = new System.Drawing.Point(56, 488);
			this.staticMissing.Name = "staticMissing";
			this.staticMissing.Size = new System.Drawing.Size(144, 24);
			this.staticMissing.TabIndex = 75;
			this.staticMissing.TextAlign = System.Drawing.ContentAlignment.TopLeft;
			// 
			// staticUnpublished
			// 
			this.staticUnpublished.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticUnpublished.LabelText = "Not published yet";
			this.staticUnpublished.Location = new System.Drawing.Point(56, 464);
			this.staticUnpublished.Name = "staticUnpublished";
			this.staticUnpublished.Size = new System.Drawing.Size(120, 24);
			this.staticUnpublished.TabIndex = 74;
			this.staticUnpublished.TextAlign = System.Drawing.ContentAlignment.TopLeft;
			// 
			// buttonSelect
			// 
			this.buttonSelect.BackColor = System.Drawing.Color.White;
			this.buttonSelect.ButtonText = "Select";
			this.buttonSelect.Location = new System.Drawing.Point(32, 512);
			this.buttonSelect.Name = "buttonSelect";
			this.buttonSelect.Size = new System.Drawing.Size(80, 24);
			this.buttonSelect.TabIndex = 82;
			this.buttonSelect.Click += new System.EventHandler(this.buttonSelect_Click);
			this.buttonSelect.DoubleClick += new System.EventHandler(this.buttonSelect_Click);
			// 
			// buttonDependencies
			// 
			this.buttonDependencies.BackColor = System.Drawing.Color.White;
			this.buttonDependencies.ButtonText = "Dependencies";
			this.buttonDependencies.Location = new System.Drawing.Point(120, 512);
			this.buttonDependencies.Name = "buttonDependencies";
			this.buttonDependencies.Size = new System.Drawing.Size(112, 24);
			this.buttonDependencies.TabIndex = 84;
			this.buttonDependencies.Click += new System.EventHandler(this.buttonDependencies_Click);
			this.buttonDependencies.DoubleClick += new System.EventHandler(this.buttonDependencies_Click);
			// 
			// groupBox1
			// 
			this.groupBox1.BackColor = System.Drawing.Color.White;
			this.groupBox1.ClosedPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
			this.groupBox1.IsOpen = true;
			this.groupBox1.IsStatic = true;
			this.groupBox1.LabelText = "Available operations";
			this.groupBox1.Location = new System.Drawing.Point(16, 88);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.OpenPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
			this.groupBox1.Size = new System.Drawing.Size(568, 456);
			this.groupBox1.TabIndex = 85;
			// 
			// groupBox2
			// 
			this.groupBox2.BackColor = System.Drawing.Color.White;
			this.groupBox2.ClosedPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
			this.groupBox2.IsOpen = true;
			this.groupBox2.IsStatic = true;
			this.groupBox2.LabelText = "Operations selected";
			this.groupBox2.Location = new System.Drawing.Point(632, 88);
			this.groupBox2.Name = "groupBox2";
			this.groupBox2.OpenPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
			this.groupBox2.Size = new System.Drawing.Size(232, 392);
			this.groupBox2.TabIndex = 86;
			// 
			// progressBar1
			// 
			this.progressBar1.BackColor = System.Drawing.Color.White;
			this.progressBar1.Location = new System.Drawing.Point(240, 516);
			this.progressBar1.Name = "progressBar1";
			this.progressBar1.Percent = 0;
			this.progressBar1.Size = new System.Drawing.Size(328, 16);
			this.progressBar1.TabIndex = 87;
			// 
			// staticHelp
			// 
			this.staticHelp.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticHelp.LabelText = "Help Text";
			this.staticHelp.Location = new System.Drawing.Point(8, 16);
			this.staticHelp.Name = "staticHelp";
			this.staticHelp.Size = new System.Drawing.Size(848, 32);
			this.staticHelp.TabIndex = 88;
			this.staticHelp.TextAlign = System.Drawing.ContentAlignment.TopLeft;
			// 
			// AddOperationsForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(882, 562);
			this.Controls.Add(this.staticHelp);
			this.Controls.Add(this.progressBar1);
			this.Controls.Add(this.buttonDependencies);
			this.Controls.Add(this.buttonSelect);
			this.Controls.Add(this.checkUnpublished);
			this.Controls.Add(this.checkMissing);
			this.Controls.Add(this.staticMissing);
			this.Controls.Add(this.staticUnpublished);
			this.Controls.Add(this.staticText3);
			this.Controls.Add(this.buttonCancel);
			this.Controls.Add(this.buttonOKPublish);
			this.Controls.Add(this.buttonRemove);
			this.Controls.Add(this.buttonClear);
			this.Controls.Add(this.buttonPublish);
			this.Controls.Add(this.staticDBLink);
			this.Controls.Add(this.textSQL);
			this.Controls.Add(this.textDBLink);
			this.Controls.Add(this.gridPublish);
			this.Controls.Add(this.gridLocal);
			this.Controls.Add(this.groupBox1);
			this.Controls.Add(this.groupBox2);
			this.Controls.Add(this.backgroundPanel1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "AddOperationsForm";
			this.Text = "Add Operation Jobs";
			((System.ComponentModel.ISupportInitialize)(this.gridLocal)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.gridPublish)).EndInit();
			this.ResumeLayout(false);

		}
		#endregion

		private void buttonSelect_Click(object sender, System.EventArgs e)
		{
			string wherelist = (string)textSQL.Text.Clone();

			string selstr = null;
			switch (AddJobType)
			{
				case SySal.Executables.OperaPublicationManager.MainForm.JobType.Compare:
				{
					if (checkMissing.Checked)
						selstr = "SELECT ID FROM TB_PROC_OPERATIONS@" + textDBLink.Text + " WHERE ID IN (SELECT ID FROM TB_PROC_OPERATIONS WHERE ID IN (" + wherelist + "))";
					else 
						selstr = "SELECT ID FROM TB_PROC_OPERATIONS WHERE ID IN (" + wherelist + ")";
				}
					break;

				case SySal.Executables.OperaPublicationManager.MainForm.JobType.Publish:
				{
					if (checkMissing.Checked)
					{
						if (checkUnpublished.Checked) selstr = "SELECT ID FROM TB_PROC_OPERATIONS@" + textDBLink.Text + " WHERE ID IN (SELECT ID FROM TB_PROC_OPERATIONS WHERE ID IN (" + wherelist + ")) AND ID NOT IN (SELECT OBJID FROM PT_OBJECTS WHERE TYPE = 'BRICK')";
						else selstr = "SELECT ID FROM TB_PROC_OPERATIONS@" + textDBLink.Text + " WHERE ID IN (SELECT ID FROM TB_PROC_OPERATIONS WHERE ID IN (" + wherelist + "))";
					}
					else 
					{
						if (checkUnpublished.Checked) selstr = "SELECT ID FROM TB_PROC_OPERATIONS WHERE ID IN (" + wherelist + ") AND ID NOT IN (SELECT OBJID FROM PT_OBJECTS WHERE TYPE = 'BRICK')";
						else selstr = "SELECT ID FROM TB_PROC_OPERATIONS WHERE ID IN (" + wherelist + ")";
					}
				}
					break;
				
				case SySal.Executables.OperaPublicationManager.MainForm.JobType.Unpublish:
				{
					if (checkMissing.Checked)
					{
						if (checkUnpublished.Checked) selstr = "SELECT ID FROM TB_PROC_OPERATIONS@" + textDBLink.Text + " WHERE ID IN (SELECT ID FROM TB_PROC_OPERATIONS WHERE ID IN (" + wherelist + ")) AND ID NOT IN (SELECT OBJID FROM PT_OBJECTS WHERE TYPE = 'BRICK')";
						else selstr = "SELECT ID FROM TB_PROC_OPERATIONS@" + textDBLink.Text + " WHERE ID IN (SELECT ID FROM TB_PROC_OPERATIONS WHERE ID IN (" + wherelist + "))";
					}
					else 
					{
						if (checkUnpublished.Checked) selstr = "SELECT ID FROM TB_PROC_OPERATIONS WHERE ID IN (" + wherelist + ") AND ID NOT IN (SELECT OBJID FROM PT_OBJECTS WHERE TYPE = 'BRICK')";
						else selstr = "SELECT ID FROM TB_PROC_OPERATIONS WHERE ID IN (" + wherelist + ")";
					}
				}
					break;

				case SySal.Executables.OperaPublicationManager.MainForm.JobType.CopyFromLink:
				{
					if (checkMissing.Checked)
					{
						if (checkUnpublished.Checked) selstr = "SELECT ID FROM TB_PROC_OPERATIONS WHERE ID IN (SELECT ID FROM TB_PROC_OPERATIONS@" + textDBLink.Text + " WHERE ID IN (" + wherelist + ")) AND ID NOT IN (SELECT OBJID FROM PT_OBJECTS WHERE TYPE = 'BRICK')";
						else selstr = "SELECT ID FROM TB_PROC_OPERATIONS WHERE ID IN (SELECT ID FROM TB_PROC_OPERATIONS@" + textDBLink.Text + " WHERE ID IN (" + wherelist + "))";
					}
					else 
					{
						if (checkUnpublished.Checked) selstr = "SELECT ID FROM TB_PROC_OPERATIONS@" + textDBLink.Text + " WHERE ID IN (" + wherelist + ") AND ID NOT IN (SELECT OBJID FROM PT_OBJECTS WHERE TYPE = 'BRICK')";
						else selstr = "SELECT ID FROM TB_PROC_OPERATIONS@" + textDBLink.Text + " WHERE ID IN (" + wherelist + ")";
					}
				}
					break;

				case SySal.Executables.OperaPublicationManager.MainForm.JobType.DeleteLocal:
				{
					selstr = "SELECT ID FROM TB_PROC_OPERATIONS WHERE ID IN (" + wherelist + ")";
				}
					break;

				case SySal.Executables.OperaPublicationManager.MainForm.JobType.CSCandidates:
				{
					if (checkMissing.Checked)
					{
						if (checkUnpublished.Checked) selstr = "SELECT ID FROM TB_PROC_OPERATIONS WHERE ID IN (SELECT ID FROM TB_PROC_OPERATIONS@" + textDBLink.Text + " WHERE ID IN (" + wherelist + ")) AND ID NOT IN (SELECT OBJID FROM PT_OBJECTS WHERE TYPE = 'BRICK')";
						else selstr = "SELECT ID FROM TB_PROC_OPERATIONS WHERE ID IN (SELECT ID FROM TB_PROC_OPERATIONS@" + textDBLink.Text + " WHERE ID IN (" + wherelist + "))";
					}
					else 
					{
						if (checkUnpublished.Checked) selstr = "SELECT ID FROM TB_PROC_OPERATIONS@" + textDBLink.Text + " WHERE ID IN (" + wherelist + ") AND ID NOT IN (SELECT OBJID FROM PT_OBJECTS WHERE TYPE = 'BRICK')";
						else selstr = "SELECT ID FROM TB_PROC_OPERATIONS@" + textDBLink.Text + " WHERE ID IN (" + wherelist + ")";
					}
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
			if (gridLocal.DataSource == null) return;
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
		/// The list of IDs of the proces operations.
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

		DependencyForm DepForm = new DependencyForm();

		private void buttonDependencies_Click(object sender, System.EventArgs e)
		{
			string baseids = "";
			int i;
			if (gridLocal.DataSource == null) return;
			int len = ((System.Data.DataTable)gridLocal.DataSource).Rows.Count;
			for (i = 0; i < len; i++)
				if (gridLocal.IsSelected(i))
				{
					if (baseids.Length > 0) baseids += ",";
					baseids += gridLocal[i,0].ToString();
				} 
			if (baseids.Length == 0) return;
			System.Data.DataSet dsvol = null;
			System.Data.DataSet dssb = null;
			System.Data.DataSet dspm1 = null;
			System.Data.DataSet dspm2 = null;
			System.Data.DataSet dscal = null;

			Cursor oldc = this.Cursor;
			this.Cursor = Cursors.WaitCursor;
			try
			{
				dsvol = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("select distinct id_processoperation from (select /*+INDEX_ASC(TB_ZONES IX_ZONES) */ distinct id_processoperation " +
					"from tb_zones where (id_eventbrick, id) in (select id_eventbrick, id_zone from tb_volume_slices where " + 
					"id_zone is not null and (id_eventbrick, id_volume) in (select id_eventbrick, id from tb_volumes where " + 
					"(id_eventbrick, id_processoperation) in (select /*+INDEX_ASC(TB_PROC_OPERATIONS IX_PROC_OPERATIONS_PARENT)*/ id_eventbrick, id from (select id, id_eventbrick from " + 
					"tb_proc_operations connect by prior id = id_parent_operation start with id in (" + baseids + ")) where " + 
					"id_eventbrick is not null )))) where id_processoperation not in (select /*+INDEX_ASC(TB_PROC_OPERATIONS IX_PROC_OPERATIONS_PARENT)*/ id from tb_proc_operations " + 
					"connect by prior id = id_parent_operation start with id in (" + baseids + "))", Conn, null).Fill(dsvol);
				dssb = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("select distinct id_processoperation from (select /*+INDEX_ASC(TB_ZONES IX_ZONES) */ distinct id_processoperation " +
					"from tb_zones where (id_eventbrick, id) in (select id_eventbrick, id_zone from tb_scanback_predictions where " + 
					"id_zone is not null and (id_eventbrick, id_path) in (select id_eventbrick, id from tb_scanback_paths where " + 
					"(id_eventbrick, id_processoperation) in (select id_eventbrick, id from (select /*+INDEX_ASC(TB_PROC_OPERATIONS IX_PROC_OPERATIONS_PARENT)*/ id, id_eventbrick from " + 
					"tb_proc_operations connect by prior id = id_parent_operation start with id in (" + baseids + ")) where " + 
					"id_eventbrick is not null )))) where id_processoperation not in (select /*+INDEX_ASC(TB_PROC_OPERATIONS IX_PROC_OPERATIONS_PARENT)*/ id from tb_proc_operations " + 
					"connect by prior id = id_parent_operation start with id in (" + baseids + "))", Conn, null).Fill(dssb);				
				dspm1 = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("select distinct id_processoperation from (select /*+INDEX_ASC(TB_ZONES IX_ZONES) */ distinct id_processoperation " +
					"from tb_zones where (id_eventbrick, id) in (select id_eventbrick, id_firstzone from tb_pattern_match " + 
					"where (id_eventbrick, id_processoperation) in ( select id_eventbrick, id from (select /*+INDEX_ASC(TB_PROC_OPERATIONS IX_PROC_OPERATIONS_PARENT)*/ id, id_eventbrick " + 
					"from tb_proc_operations connect by prior id = id_parent_operation start with id in (" + baseids + 
					")) where id_eventbrick is not null ))) where id_processoperation not in (select /*+INDEX_ASC(TB_PROC_OPERATIONS IX_PROC_OPERATIONS_PARENT)*/ id from tb_proc_operations " +
					"connect by prior id = id_parent_operation start with id in (" + baseids + "))", Conn, null).Fill(dspm1);
				dspm2 = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("select distinct id_processoperation from (select /*+INDEX_ASC(TB_ZONES IX_ZONES) */ distinct id_processoperation " +
					"from tb_zones where (id_eventbrick, id) in (select id_eventbrick, id_secondzone from tb_pattern_match " + 
					"where (id_eventbrick, id_processoperation) in ( select id_eventbrick, id from (select /*+INDEX_ASC(TB_PROC_OPERATIONS IX_PROC_OPERATIONS_PARENT)*/ id, id_eventbrick " + 
					"from tb_proc_operations connect by prior id = id_parent_operation start with id in (" + baseids + 
					")) where id_eventbrick is not null ))) where id_processoperation not in (select /*+INDEX_ASC(TB_PROC_OPERATIONS IX_PROC_OPERATIONS_PARENT)*/ id from tb_proc_operations " +
					"connect by prior id = id_parent_operation start with id in (" + baseids + "))", Conn, null).Fill(dspm2);
				dscal = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("select id_calibration_operation from (select /*+INDEX_ASC(TB_PROC_OPERATIONS IX_PROC_OPERATIONS_PARENT)*/ id_calibration_operation from " +
					"tb_proc_operations connect by prior id = id_parent_operation start with id in (" + baseids + ")) where id_calibration_operation is not null and id_calibration_operation " + 
					"not in (select /*+INDEX_ASC(TB_PROC_OPERATIONS IX_PROC_OPERATIONS_PARENT)*/ id from tb_proc_operations " +
					"connect by prior id = id_parent_operation start with id in (" + baseids + "))", Conn, null).Fill(dscal);
				System.Data.DataTable deps = new System.Data.DataTable();
				deps.Columns.Add("Process operation ID");
				deps.Columns.Add("Type of dependency");
				foreach (System.Data.DataRow dr in dsvol.Tables[0].Rows)
				{
					System.Data.DataRow nr = deps.NewRow();
					nr[0] = dr[0].ToString();
					nr[1] = "Volume slice";
					deps.Rows.Add(nr);
				}
				foreach (System.Data.DataRow dr in dssb.Tables[0].Rows)
				{
					System.Data.DataRow nr = deps.NewRow();
					nr[0] = dr[0].ToString();
					nr[1] = "Scanback prediction";
					deps.Rows.Add(nr);
				}
				foreach (System.Data.DataRow dr in dspm1.Tables[0].Rows)
				{
					System.Data.DataRow nr = deps.NewRow();
					nr[0] = dr[0].ToString();
					nr[1] = "Pattern match, first zone";
					deps.Rows.Add(nr);
				}
				foreach (System.Data.DataRow dr in dspm2.Tables[0].Rows)
				{
					System.Data.DataRow nr = deps.NewRow();
					nr[0] = dr[0].ToString();
					nr[1] = "Pattern match, second zone";
					deps.Rows.Add(nr);
				}
				foreach (System.Data.DataRow dr in dscal.Tables[0].Rows)
				{
					System.Data.DataRow nr = deps.NewRow();
					nr[0] = dr[0].ToString();
					nr[1] = "Plate calibration";
					deps.Rows.Add(nr);
				}
				if (deps.Rows.Count == 0) MessageBox.Show("No dependencies found.", "Dependency report", MessageBoxButtons.OK, MessageBoxIcon.Information);					
				else DepForm.ShowDialog(deps);

			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error in dependency walk", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
			this.Cursor = oldc;
		}

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
				case SySal.Executables.OperaPublicationManager.MainForm.JobType.Compare:		this.Text = "Add Operation 'COMPARE' Jobs"; staticDBLink.Visible = true; textDBLink.Visible = true; checkUnpublished.Visible = false; staticUnpublished.Visible = false; checkMissing.Visible = true; staticMissing.Visible = true; 
					staticHelp.LabelText = "Please select the operations whose local data you want to compare with data stored on the remote DB.\r\nComparison involves both signature checking and row count comparison."; break;

				case SySal.Executables.OperaPublicationManager.MainForm.JobType.Publish:		this.Text = "Add Operation 'PUBLISH' Jobs"; staticDBLink.Visible = true; textDBLink.Visible = true; checkUnpublished.Visible = true; staticUnpublished.Visible = true; checkMissing.Visible = true; staticMissing.Visible = true; 
					staticHelp.LabelText = "Please select the operations whose local data you want to upload to the remote DB."; break;

				case SySal.Executables.OperaPublicationManager.MainForm.JobType.Unpublish:		this.Text = "Add Operation 'UNPUBLISH' Jobs"; staticDBLink.Visible = true; textDBLink.Visible = true; checkUnpublished.Visible = true; staticUnpublished.Visible = true; checkMissing.Visible = true; staticMissing.Visible = true; 
					staticHelp.LabelText = "Please select the operations whose local data you had published and you want now to remove\r\nfrom the remote DB."; break;

				case SySal.Executables.OperaPublicationManager.MainForm.JobType.CopyFromLink:	this.Text = "Add Operation 'COPY FROM LINK' Jobs"; staticDBLink.Visible = true; textDBLink.Visible = true; checkUnpublished.Visible = true; staticUnpublished.Visible = true; checkMissing.Visible = true; staticMissing.Visible = true; 
					staticHelp.LabelText = "Please select the operations whose data you want to download from the remote DB."; break;

				case SySal.Executables.OperaPublicationManager.MainForm.JobType.DeleteLocal:	this.Text = "Add Operation 'DELETE LOCAL' Jobs"; staticDBLink.Visible = false; textDBLink.Visible = false; checkUnpublished.Visible = false; staticUnpublished.Visible = false; checkMissing.Visible = false; staticMissing.Visible = false;
					staticHelp.LabelText = "Please select the operations whose data you want to delete from the local DB."; break;

				case SySal.Executables.OperaPublicationManager.MainForm.JobType.CSCandidates:	this.Text = "Add Operation 'GET CS CANDIDATES' Jobs"; staticDBLink.Visible = true; textDBLink.Visible = true; checkUnpublished.Visible = true; staticUnpublished.Visible = true; checkMissing.Visible = true; staticMissing.Visible = true; 
					staticHelp.LabelText = "Please select the CS operations whose candidates you want to download from the remote DB."; break;

			}
			try
			{
				res = ShowDialog();
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error executing \"Add operations\" form", MessageBoxButtons.OK, MessageBoxIcon.Error);
				res = DialogResult.Cancel;
			}
			Conn = null;			
			return res;
		}

	}
}
