using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;

namespace SySal.Executables.EasySite
{
	/// <summary>
	/// EasySite - GUI tool for easy management of Computing Infrastructure sites.
	/// </summary>
	/// <remarks>
	/// <para>
	/// EasySite always loads its OperaDB and Computing Infrastructure credentials from the user credential record. If error messages are displayed, close EasySite, enter your credentials by <see cref="SySal.Executables.OperaDbGUILogin.MainForm">OperaDbGUILogin</see> or <see cref="SySal.Executables.OperaDbTextLogin.Exe">OperaDbTextLogin</see> and open EasySite again.
	/// </para>
	/// <para>
	/// EasySite is based on two tables that do not exist in the standard OperaDB: LZ_SITEVARS and LZ_MACHINEVARS. 
	/// Upon launch, EasySite checks their existence. If they are not found, they are created. (<b>NOTICE: the user must have the DB privileges/permissions to create tables in the OPERA schema.</b>).	
	/// </para>
	/// <para>
	/// The first time EasySite runs, the current site should be set. On doing so, the LZ_SITEVARS table is filled with default values for relevant parameters of BatchManagers and DataProcessingServers.
	/// </para>
	/// <para>
	/// The following items can be administered:
	/// <list type="table">
	/// <listheader><term>Item</term><description>Actions</description></listheader>
	/// <item><term>Site</term><description>site-wide default parameters (i.e. parameters that apply to all machines unless specifically overridden); e.g. the Scratch directory, the ExeRepository, the location of Task Progress files, etc.</description></item>
	/// <item><term>Machines</term><description>this view allows you to add/edit/delete machine records in the DB, and to set specific overrides for parameters (e.g. the MachinePowerClass for a DataProcessingServer, which is machine-dependent).</description></item>
	/// <item><term>Users</term><description>this view allows you to add/edit/delete user records and permissions.</description></item>
	/// <item><term>Program Settings</term><description>this view allows you to add or delete program settings. It is important to specify also the correct driver type (by the list in the lower right corner).</description></item>
	/// </list>
	/// </para>
	/// </remarks>
	public class MainForm : System.Windows.Forms.Form
	{
		SySal.OperaDb.OperaDbConnection Conn;

		private System.Windows.Forms.TabControl GroupTab;
		private System.Windows.Forms.TabPage tabPage1;
		private System.Windows.Forms.TabPage tabPage2;
		private System.Windows.Forms.TabPage tabPage3;
		private System.Windows.Forms.TabPage tabPage4;
		private System.Windows.Forms.ListView SiteList;
		private System.Windows.Forms.Button SiteMarkCurrent;
		private System.Windows.Forms.ColumnHeader columnHeader1;
		private System.Windows.Forms.ColumnHeader columnHeader2;
		private System.Windows.Forms.ColumnHeader columnHeader3;
		private System.Windows.Forms.ColumnHeader columnHeader4;
		private System.Windows.Forms.ListView SiteVars;
		private System.Windows.Forms.ColumnHeader columnHeader5;
		private System.Windows.Forms.ColumnHeader columnHeader6;
		private System.Windows.Forms.TextBox SiteVarName;
		private System.Windows.Forms.TextBox SiteVarValue;
		private System.Windows.Forms.Button SiteVarSet;
		private System.Windows.Forms.Button SiteVarDel;
		private System.Windows.Forms.ColumnHeader columnHeader7;
		private System.Windows.Forms.ColumnHeader columnHeader8;
		private System.Windows.Forms.ColumnHeader columnHeader9;
		private System.Windows.Forms.ColumnHeader columnHeader10;
		private System.Windows.Forms.ColumnHeader columnHeader11;
		private System.Windows.Forms.ColumnHeader columnHeader12;
		private System.Windows.Forms.ColumnHeader columnHeader13;
		private System.Windows.Forms.ColumnHeader columnHeader14;
		private System.Windows.Forms.ColumnHeader columnHeader15;
		private System.Windows.Forms.ColumnHeader columnHeader16;
		private System.Windows.Forms.ColumnHeader columnHeader17;
		private System.Windows.Forms.ColumnHeader columnHeader18;
		private System.Windows.Forms.ColumnHeader columnHeader19;
		private System.Windows.Forms.ColumnHeader columnHeader20;
		private System.Windows.Forms.ColumnHeader columnHeader21;
		private System.Windows.Forms.ColumnHeader columnHeader22;
		private System.Windows.Forms.ColumnHeader columnHeader23;
		private System.Windows.Forms.ColumnHeader columnHeader24;
		private System.Windows.Forms.ColumnHeader columnHeader25;
		private System.Windows.Forms.ColumnHeader columnHeader26;
		private System.Windows.Forms.ColumnHeader columnHeader27;
		private System.Windows.Forms.ColumnHeader columnHeader28;
		private System.Windows.Forms.ColumnHeader columnHeader29;
		private System.Windows.Forms.ColumnHeader columnHeader30;
		private System.Windows.Forms.ColumnHeader columnHeader31;
		private System.Windows.Forms.ColumnHeader columnHeader32;
		private System.Windows.Forms.ListView MachineList;
		private System.Windows.Forms.ListView MachineVars;
		private System.Windows.Forms.TextBox MachineVarName;
		private System.Windows.Forms.TextBox MachineVarValue;
		private System.Windows.Forms.Button MachineVarDel;
		private System.Windows.Forms.Button MachineVarSet;
		private System.Windows.Forms.TextBox MachineName;
		private System.Windows.Forms.TextBox MachineAddr;
		private System.Windows.Forms.Button MachineSet;
		private System.Windows.Forms.Button MachineDel;
		private System.Windows.Forms.TextBox MachineFunc;
		private System.Windows.Forms.ColumnHeader columnHeader33;
		private System.Windows.Forms.ColumnHeader columnHeader34;
		private System.Windows.Forms.ColumnHeader columnHeader35;
		private System.Windows.Forms.ColumnHeader columnHeader36;
		private System.Windows.Forms.ColumnHeader columnHeader37;
		private System.Windows.Forms.Button MachineAdd;
		private System.Windows.Forms.Button ProgSetAdd;
		private System.Windows.Forms.Button ProgSetDel;
		private System.Windows.Forms.TextBox ProgSetAuthor;
		private System.Windows.Forms.ComboBox ProgSetLevel;
		private System.Windows.Forms.TextBox ProgSetExecutable;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox ProgSetDescription;
		private System.Windows.Forms.ColumnHeader columnHeader38;
		private System.Windows.Forms.ColumnHeader columnHeader39;
		private System.Windows.Forms.ColumnHeader columnHeader40;
		private System.Windows.Forms.ListView ProgSetList;
		private System.Windows.Forms.TextBox ProgSetSettings;
		private System.Windows.Forms.ListView UserList;
		private System.Windows.Forms.TextBox UserUsername;
		private System.Windows.Forms.TextBox UserPassword;
		private System.Windows.Forms.TextBox UserName;
		private System.Windows.Forms.TextBox UserSurname;
		private System.Windows.Forms.TextBox UserInstitution;
		private System.Windows.Forms.TextBox UserEmail;
		private System.Windows.Forms.TextBox UserAddress;
		private System.Windows.Forms.TextBox UserPhone;
		private System.Windows.Forms.TextBox UserPermissions;
		private System.Windows.Forms.Button UserAdd;
		private System.Windows.Forms.Button UserSet;
		private System.Windows.Forms.Button UserDel;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.ColumnHeader columnHeader41;
		private System.Windows.Forms.ColumnHeader columnHeader42;
		private System.Windows.Forms.ColumnHeader columnHeader43;
		private System.Windows.Forms.ColumnHeader columnHeader44;
		private System.Windows.Forms.ColumnHeader columnHeader45;
		private System.Windows.Forms.ColumnHeader columnHeader46;
		private System.Windows.Forms.ColumnHeader columnHeader47;
		private System.Windows.Forms.ColumnHeader columnHeader48;
        private ColumnHeader columnHeader49;
        private CheckedListBox ProgSetCheckList;
        private Label label4;
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
            this.GroupTab = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.SiteVarDel = new System.Windows.Forms.Button();
            this.SiteVarSet = new System.Windows.Forms.Button();
            this.SiteVarValue = new System.Windows.Forms.TextBox();
            this.SiteVarName = new System.Windows.Forms.TextBox();
            this.SiteVars = new System.Windows.Forms.ListView();
            this.columnHeader5 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader6 = new System.Windows.Forms.ColumnHeader();
            this.SiteMarkCurrent = new System.Windows.Forms.Button();
            this.SiteList = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader4 = new System.Windows.Forms.ColumnHeader();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.MachineAdd = new System.Windows.Forms.Button();
            this.MachineFunc = new System.Windows.Forms.TextBox();
            this.MachineSet = new System.Windows.Forms.Button();
            this.MachineDel = new System.Windows.Forms.Button();
            this.MachineAddr = new System.Windows.Forms.TextBox();
            this.MachineName = new System.Windows.Forms.TextBox();
            this.MachineVarSet = new System.Windows.Forms.Button();
            this.MachineVarDel = new System.Windows.Forms.Button();
            this.MachineVarValue = new System.Windows.Forms.TextBox();
            this.MachineVarName = new System.Windows.Forms.TextBox();
            this.MachineVars = new System.Windows.Forms.ListView();
            this.columnHeader36 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader37 = new System.Windows.Forms.ColumnHeader();
            this.MachineList = new System.Windows.Forms.ListView();
            this.columnHeader33 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader34 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader35 = new System.Windows.Forms.ColumnHeader();
            this.tabPage3 = new System.Windows.Forms.TabPage();
            this.label3 = new System.Windows.Forms.Label();
            this.UserDel = new System.Windows.Forms.Button();
            this.UserSet = new System.Windows.Forms.Button();
            this.UserAdd = new System.Windows.Forms.Button();
            this.UserPermissions = new System.Windows.Forms.TextBox();
            this.UserPhone = new System.Windows.Forms.TextBox();
            this.UserAddress = new System.Windows.Forms.TextBox();
            this.UserEmail = new System.Windows.Forms.TextBox();
            this.UserInstitution = new System.Windows.Forms.TextBox();
            this.UserSurname = new System.Windows.Forms.TextBox();
            this.UserName = new System.Windows.Forms.TextBox();
            this.UserPassword = new System.Windows.Forms.TextBox();
            this.UserUsername = new System.Windows.Forms.TextBox();
            this.UserList = new System.Windows.Forms.ListView();
            this.columnHeader41 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader42 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader43 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader44 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader45 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader46 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader47 = new System.Windows.Forms.ColumnHeader();
            this.tabPage4 = new System.Windows.Forms.TabPage();
            this.label2 = new System.Windows.Forms.Label();
            this.ProgSetDescription = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.ProgSetExecutable = new System.Windows.Forms.TextBox();
            this.ProgSetLevel = new System.Windows.Forms.ComboBox();
            this.ProgSetSettings = new System.Windows.Forms.TextBox();
            this.ProgSetAuthor = new System.Windows.Forms.TextBox();
            this.ProgSetDel = new System.Windows.Forms.Button();
            this.ProgSetAdd = new System.Windows.Forms.Button();
            this.ProgSetList = new System.Windows.Forms.ListView();
            this.columnHeader38 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader39 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader40 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader48 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader7 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader8 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader9 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader10 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader11 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader12 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader13 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader14 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader15 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader16 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader17 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader18 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader19 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader20 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader21 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader22 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader23 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader24 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader25 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader26 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader27 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader28 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader29 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader30 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader31 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader32 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader49 = new System.Windows.Forms.ColumnHeader();
            this.label4 = new System.Windows.Forms.Label();
            this.ProgSetCheckList = new System.Windows.Forms.CheckedListBox();
            this.GroupTab.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.tabPage2.SuspendLayout();
            this.tabPage3.SuspendLayout();
            this.tabPage4.SuspendLayout();
            this.SuspendLayout();
            // 
            // GroupTab
            // 
            this.GroupTab.Controls.Add(this.tabPage1);
            this.GroupTab.Controls.Add(this.tabPage2);
            this.GroupTab.Controls.Add(this.tabPage3);
            this.GroupTab.Controls.Add(this.tabPage4);
            this.GroupTab.Location = new System.Drawing.Point(8, 8);
            this.GroupTab.Name = "GroupTab";
            this.GroupTab.SelectedIndex = 0;
            this.GroupTab.Size = new System.Drawing.Size(816, 430);
            this.GroupTab.TabIndex = 0;
            // 
            // tabPage1
            // 
            this.tabPage1.Controls.Add(this.SiteVarDel);
            this.tabPage1.Controls.Add(this.SiteVarSet);
            this.tabPage1.Controls.Add(this.SiteVarValue);
            this.tabPage1.Controls.Add(this.SiteVarName);
            this.tabPage1.Controls.Add(this.SiteVars);
            this.tabPage1.Controls.Add(this.SiteMarkCurrent);
            this.tabPage1.Controls.Add(this.SiteList);
            this.tabPage1.Location = new System.Drawing.Point(4, 22);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Size = new System.Drawing.Size(808, 404);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "Site";
            // 
            // SiteVarDel
            // 
            this.SiteVarDel.Location = new System.Drawing.Point(760, 372);
            this.SiteVarDel.Name = "SiteVarDel";
            this.SiteVarDel.Size = new System.Drawing.Size(40, 24);
            this.SiteVarDel.TabIndex = 7;
            this.SiteVarDel.Text = "Del";
            this.SiteVarDel.Click += new System.EventHandler(this.SiteVarDel_Click);
            // 
            // SiteVarSet
            // 
            this.SiteVarSet.Location = new System.Drawing.Point(712, 372);
            this.SiteVarSet.Name = "SiteVarSet";
            this.SiteVarSet.Size = new System.Drawing.Size(40, 24);
            this.SiteVarSet.TabIndex = 6;
            this.SiteVarSet.Text = "Set";
            this.SiteVarSet.Click += new System.EventHandler(this.SiteVarSet_Click);
            // 
            // SiteVarValue
            // 
            this.SiteVarValue.Location = new System.Drawing.Point(518, 346);
            this.SiteVarValue.Name = "SiteVarValue";
            this.SiteVarValue.Size = new System.Drawing.Size(280, 20);
            this.SiteVarValue.TabIndex = 4;
            // 
            // SiteVarName
            // 
            this.SiteVarName.Location = new System.Drawing.Point(414, 346);
            this.SiteVarName.Name = "SiteVarName";
            this.SiteVarName.Size = new System.Drawing.Size(96, 20);
            this.SiteVarName.TabIndex = 3;
            // 
            // SiteVars
            // 
            this.SiteVars.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader5,
            this.columnHeader6});
            this.SiteVars.FullRowSelect = true;
            this.SiteVars.GridLines = true;
            this.SiteVars.HideSelection = false;
            this.SiteVars.Location = new System.Drawing.Point(416, 8);
            this.SiteVars.Name = "SiteVars";
            this.SiteVars.Size = new System.Drawing.Size(384, 332);
            this.SiteVars.Sorting = System.Windows.Forms.SortOrder.Ascending;
            this.SiteVars.TabIndex = 2;
            this.SiteVars.UseCompatibleStateImageBehavior = false;
            this.SiteVars.View = System.Windows.Forms.View.Details;
            this.SiteVars.SelectedIndexChanged += new System.EventHandler(this.OnSiteVarSelChanged);
            // 
            // columnHeader5
            // 
            this.columnHeader5.Text = "Name";
            // 
            // columnHeader6
            // 
            this.columnHeader6.Text = "Value";
            // 
            // SiteMarkCurrent
            // 
            this.SiteMarkCurrent.Location = new System.Drawing.Point(8, 372);
            this.SiteMarkCurrent.Name = "SiteMarkCurrent";
            this.SiteMarkCurrent.Size = new System.Drawing.Size(88, 24);
            this.SiteMarkCurrent.TabIndex = 1;
            this.SiteMarkCurrent.Text = "Set current site";
            this.SiteMarkCurrent.Click += new System.EventHandler(this.SiteMarkCurrent_Click);
            // 
            // SiteList
            // 
            this.SiteList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2,
            this.columnHeader3,
            this.columnHeader4});
            this.SiteList.FullRowSelect = true;
            this.SiteList.GridLines = true;
            this.SiteList.HideSelection = false;
            this.SiteList.Location = new System.Drawing.Point(8, 8);
            this.SiteList.Name = "SiteList";
            this.SiteList.Size = new System.Drawing.Size(400, 358);
            this.SiteList.TabIndex = 0;
            this.SiteList.UseCompatibleStateImageBehavior = false;
            this.SiteList.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Name";
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Longitude";
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "Latitude";
            // 
            // columnHeader4
            // 
            this.columnHeader4.Text = "Local Time Fuse";
            // 
            // tabPage2
            // 
            this.tabPage2.Controls.Add(this.MachineAdd);
            this.tabPage2.Controls.Add(this.MachineFunc);
            this.tabPage2.Controls.Add(this.MachineSet);
            this.tabPage2.Controls.Add(this.MachineDel);
            this.tabPage2.Controls.Add(this.MachineAddr);
            this.tabPage2.Controls.Add(this.MachineName);
            this.tabPage2.Controls.Add(this.MachineVarSet);
            this.tabPage2.Controls.Add(this.MachineVarDel);
            this.tabPage2.Controls.Add(this.MachineVarValue);
            this.tabPage2.Controls.Add(this.MachineVarName);
            this.tabPage2.Controls.Add(this.MachineVars);
            this.tabPage2.Controls.Add(this.MachineList);
            this.tabPage2.Location = new System.Drawing.Point(4, 22);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Size = new System.Drawing.Size(808, 404);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "Machines";
            // 
            // MachineAdd
            // 
            this.MachineAdd.Location = new System.Drawing.Point(8, 372);
            this.MachineAdd.Name = "MachineAdd";
            this.MachineAdd.Size = new System.Drawing.Size(40, 24);
            this.MachineAdd.TabIndex = 11;
            this.MachineAdd.Text = "Add";
            this.MachineAdd.Click += new System.EventHandler(this.MachineAdd_Click);
            // 
            // MachineFunc
            // 
            this.MachineFunc.Location = new System.Drawing.Point(344, 346);
            this.MachineFunc.Name = "MachineFunc";
            this.MachineFunc.Size = new System.Drawing.Size(64, 20);
            this.MachineFunc.TabIndex = 10;
            // 
            // MachineSet
            // 
            this.MachineSet.Location = new System.Drawing.Point(56, 372);
            this.MachineSet.Name = "MachineSet";
            this.MachineSet.Size = new System.Drawing.Size(40, 24);
            this.MachineSet.TabIndex = 9;
            this.MachineSet.Text = "Set";
            this.MachineSet.Click += new System.EventHandler(this.MachineSet_Click);
            // 
            // MachineDel
            // 
            this.MachineDel.Location = new System.Drawing.Point(104, 372);
            this.MachineDel.Name = "MachineDel";
            this.MachineDel.Size = new System.Drawing.Size(40, 24);
            this.MachineDel.TabIndex = 8;
            this.MachineDel.Text = "Del";
            this.MachineDel.Click += new System.EventHandler(this.MachineDel_Click);
            // 
            // MachineAddr
            // 
            this.MachineAddr.Location = new System.Drawing.Point(176, 346);
            this.MachineAddr.Name = "MachineAddr";
            this.MachineAddr.Size = new System.Drawing.Size(160, 20);
            this.MachineAddr.TabIndex = 7;
            // 
            // MachineName
            // 
            this.MachineName.Location = new System.Drawing.Point(8, 346);
            this.MachineName.Name = "MachineName";
            this.MachineName.Size = new System.Drawing.Size(160, 20);
            this.MachineName.TabIndex = 6;
            // 
            // MachineVarSet
            // 
            this.MachineVarSet.Location = new System.Drawing.Point(712, 372);
            this.MachineVarSet.Name = "MachineVarSet";
            this.MachineVarSet.Size = new System.Drawing.Size(40, 24);
            this.MachineVarSet.TabIndex = 5;
            this.MachineVarSet.Text = "Set";
            this.MachineVarSet.Click += new System.EventHandler(this.MachineVarSet_Click);
            // 
            // MachineVarDel
            // 
            this.MachineVarDel.Location = new System.Drawing.Point(760, 372);
            this.MachineVarDel.Name = "MachineVarDel";
            this.MachineVarDel.Size = new System.Drawing.Size(40, 24);
            this.MachineVarDel.TabIndex = 4;
            this.MachineVarDel.Text = "Del";
            this.MachineVarDel.Click += new System.EventHandler(this.MachineVarDel_Click);
            // 
            // MachineVarValue
            // 
            this.MachineVarValue.Location = new System.Drawing.Point(520, 346);
            this.MachineVarValue.Name = "MachineVarValue";
            this.MachineVarValue.Size = new System.Drawing.Size(280, 20);
            this.MachineVarValue.TabIndex = 3;
            // 
            // MachineVarName
            // 
            this.MachineVarName.Location = new System.Drawing.Point(416, 346);
            this.MachineVarName.Name = "MachineVarName";
            this.MachineVarName.Size = new System.Drawing.Size(96, 20);
            this.MachineVarName.TabIndex = 2;
            // 
            // MachineVars
            // 
            this.MachineVars.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader36,
            this.columnHeader37});
            this.MachineVars.FullRowSelect = true;
            this.MachineVars.GridLines = true;
            this.MachineVars.HideSelection = false;
            this.MachineVars.Location = new System.Drawing.Point(416, 8);
            this.MachineVars.Name = "MachineVars";
            this.MachineVars.Size = new System.Drawing.Size(384, 332);
            this.MachineVars.TabIndex = 1;
            this.MachineVars.UseCompatibleStateImageBehavior = false;
            this.MachineVars.View = System.Windows.Forms.View.Details;
            this.MachineVars.SelectedIndexChanged += new System.EventHandler(this.OnMachineVarSelChanged);
            // 
            // columnHeader36
            // 
            this.columnHeader36.Text = "Name";
            // 
            // columnHeader37
            // 
            this.columnHeader37.Text = "Value";
            // 
            // MachineList
            // 
            this.MachineList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader33,
            this.columnHeader34,
            this.columnHeader35});
            this.MachineList.FullRowSelect = true;
            this.MachineList.GridLines = true;
            this.MachineList.HideSelection = false;
            this.MachineList.Location = new System.Drawing.Point(8, 8);
            this.MachineList.Name = "MachineList";
            this.MachineList.Size = new System.Drawing.Size(400, 332);
            this.MachineList.TabIndex = 0;
            this.MachineList.UseCompatibleStateImageBehavior = false;
            this.MachineList.View = System.Windows.Forms.View.Details;
            this.MachineList.SelectedIndexChanged += new System.EventHandler(this.OnMachineSelChanged);
            // 
            // columnHeader33
            // 
            this.columnHeader33.Text = "Name";
            // 
            // columnHeader34
            // 
            this.columnHeader34.Text = "Address";
            // 
            // columnHeader35
            // 
            this.columnHeader35.Text = "Functions";
            // 
            // tabPage3
            // 
            this.tabPage3.Controls.Add(this.label3);
            this.tabPage3.Controls.Add(this.UserDel);
            this.tabPage3.Controls.Add(this.UserSet);
            this.tabPage3.Controls.Add(this.UserAdd);
            this.tabPage3.Controls.Add(this.UserPermissions);
            this.tabPage3.Controls.Add(this.UserPhone);
            this.tabPage3.Controls.Add(this.UserAddress);
            this.tabPage3.Controls.Add(this.UserEmail);
            this.tabPage3.Controls.Add(this.UserInstitution);
            this.tabPage3.Controls.Add(this.UserSurname);
            this.tabPage3.Controls.Add(this.UserName);
            this.tabPage3.Controls.Add(this.UserPassword);
            this.tabPage3.Controls.Add(this.UserUsername);
            this.tabPage3.Controls.Add(this.UserList);
            this.tabPage3.Location = new System.Drawing.Point(4, 22);
            this.tabPage3.Name = "tabPage3";
            this.tabPage3.Size = new System.Drawing.Size(808, 404);
            this.tabPage3.TabIndex = 2;
            this.tabPage3.Text = "Users";
            // 
            // label3
            // 
            this.label3.Location = new System.Drawing.Point(623, 372);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(112, 24);
            this.label3.TabIndex = 13;
            this.label3.Text = "Permissions";
            this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // UserDel
            // 
            this.UserDel.Location = new System.Drawing.Point(119, 372);
            this.UserDel.Name = "UserDel";
            this.UserDel.Size = new System.Drawing.Size(48, 24);
            this.UserDel.TabIndex = 12;
            this.UserDel.Text = "Del";
            this.UserDel.Click += new System.EventHandler(this.UserDel_Click);
            // 
            // UserSet
            // 
            this.UserSet.Location = new System.Drawing.Point(63, 372);
            this.UserSet.Name = "UserSet";
            this.UserSet.Size = new System.Drawing.Size(48, 24);
            this.UserSet.TabIndex = 11;
            this.UserSet.Text = "Set";
            this.UserSet.Click += new System.EventHandler(this.UserSet_Click);
            // 
            // UserAdd
            // 
            this.UserAdd.Location = new System.Drawing.Point(7, 372);
            this.UserAdd.Name = "UserAdd";
            this.UserAdd.Size = new System.Drawing.Size(48, 24);
            this.UserAdd.TabIndex = 10;
            this.UserAdd.Text = "Add";
            this.UserAdd.Click += new System.EventHandler(this.UserAdd_Click);
            // 
            // UserPermissions
            // 
            this.UserPermissions.Location = new System.Drawing.Point(743, 372);
            this.UserPermissions.Name = "UserPermissions";
            this.UserPermissions.Size = new System.Drawing.Size(56, 20);
            this.UserPermissions.TabIndex = 9;
            // 
            // UserPhone
            // 
            this.UserPhone.Location = new System.Drawing.Point(743, 346);
            this.UserPhone.Name = "UserPhone";
            this.UserPhone.Size = new System.Drawing.Size(56, 20);
            this.UserPhone.TabIndex = 8;
            // 
            // UserAddress
            // 
            this.UserAddress.Location = new System.Drawing.Point(639, 346);
            this.UserAddress.Name = "UserAddress";
            this.UserAddress.Size = new System.Drawing.Size(96, 20);
            this.UserAddress.TabIndex = 7;
            // 
            // UserEmail
            // 
            this.UserEmail.Location = new System.Drawing.Point(527, 346);
            this.UserEmail.Name = "UserEmail";
            this.UserEmail.Size = new System.Drawing.Size(104, 20);
            this.UserEmail.TabIndex = 6;
            // 
            // UserInstitution
            // 
            this.UserInstitution.Location = new System.Drawing.Point(407, 346);
            this.UserInstitution.Name = "UserInstitution";
            this.UserInstitution.Size = new System.Drawing.Size(112, 20);
            this.UserInstitution.TabIndex = 5;
            // 
            // UserSurname
            // 
            this.UserSurname.Location = new System.Drawing.Point(295, 346);
            this.UserSurname.Name = "UserSurname";
            this.UserSurname.Size = new System.Drawing.Size(104, 20);
            this.UserSurname.TabIndex = 4;
            // 
            // UserName
            // 
            this.UserName.Location = new System.Drawing.Point(183, 346);
            this.UserName.Name = "UserName";
            this.UserName.Size = new System.Drawing.Size(104, 20);
            this.UserName.TabIndex = 3;
            // 
            // UserPassword
            // 
            this.UserPassword.Location = new System.Drawing.Point(71, 346);
            this.UserPassword.Name = "UserPassword";
            this.UserPassword.Size = new System.Drawing.Size(104, 20);
            this.UserPassword.TabIndex = 2;
            // 
            // UserUsername
            // 
            this.UserUsername.Location = new System.Drawing.Point(7, 346);
            this.UserUsername.Name = "UserUsername";
            this.UserUsername.Size = new System.Drawing.Size(56, 20);
            this.UserUsername.TabIndex = 1;
            // 
            // UserList
            // 
            this.UserList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader41,
            this.columnHeader42,
            this.columnHeader43,
            this.columnHeader44,
            this.columnHeader45,
            this.columnHeader46,
            this.columnHeader47});
            this.UserList.FullRowSelect = true;
            this.UserList.GridLines = true;
            this.UserList.HideSelection = false;
            this.UserList.Location = new System.Drawing.Point(8, 8);
            this.UserList.Name = "UserList";
            this.UserList.Size = new System.Drawing.Size(792, 332);
            this.UserList.TabIndex = 0;
            this.UserList.UseCompatibleStateImageBehavior = false;
            this.UserList.View = System.Windows.Forms.View.Details;
            this.UserList.SelectedIndexChanged += new System.EventHandler(this.OnUserSelChanged);
            // 
            // columnHeader41
            // 
            this.columnHeader41.Text = "Username";
            // 
            // columnHeader42
            // 
            this.columnHeader42.Text = "Name";
            // 
            // columnHeader43
            // 
            this.columnHeader43.Text = "Surname";
            // 
            // columnHeader44
            // 
            this.columnHeader44.Text = "Institution";
            // 
            // columnHeader45
            // 
            this.columnHeader45.Text = "Email";
            // 
            // columnHeader46
            // 
            this.columnHeader46.Text = "Address";
            // 
            // columnHeader47
            // 
            this.columnHeader47.Text = "Phone";
            // 
            // tabPage4
            // 
            this.tabPage4.Controls.Add(this.ProgSetCheckList);
            this.tabPage4.Controls.Add(this.label4);
            this.tabPage4.Controls.Add(this.label2);
            this.tabPage4.Controls.Add(this.ProgSetDescription);
            this.tabPage4.Controls.Add(this.label1);
            this.tabPage4.Controls.Add(this.ProgSetExecutable);
            this.tabPage4.Controls.Add(this.ProgSetLevel);
            this.tabPage4.Controls.Add(this.ProgSetSettings);
            this.tabPage4.Controls.Add(this.ProgSetAuthor);
            this.tabPage4.Controls.Add(this.ProgSetDel);
            this.tabPage4.Controls.Add(this.ProgSetAdd);
            this.tabPage4.Controls.Add(this.ProgSetList);
            this.tabPage4.Location = new System.Drawing.Point(4, 22);
            this.tabPage4.Name = "tabPage4";
            this.tabPage4.Size = new System.Drawing.Size(808, 404);
            this.tabPage4.TabIndex = 3;
            this.tabPage4.Text = "Program settings";
            // 
            // label2
            // 
            this.label2.Location = new System.Drawing.Point(520, 56);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(40, 24);
            this.label2.TabIndex = 9;
            this.label2.Text = "Desc";
            this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // ProgSetDescription
            // 
            this.ProgSetDescription.Location = new System.Drawing.Point(560, 56);
            this.ProgSetDescription.Name = "ProgSetDescription";
            this.ProgSetDescription.Size = new System.Drawing.Size(242, 20);
            this.ProgSetDescription.TabIndex = 8;
            // 
            // label1
            // 
            this.label1.Location = new System.Drawing.Point(520, 32);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(40, 24);
            this.label1.TabIndex = 7;
            this.label1.Text = "Exe";
            this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // ProgSetExecutable
            // 
            this.ProgSetExecutable.Location = new System.Drawing.Point(560, 32);
            this.ProgSetExecutable.Name = "ProgSetExecutable";
            this.ProgSetExecutable.Size = new System.Drawing.Size(242, 20);
            this.ProgSetExecutable.TabIndex = 6;
            // 
            // ProgSetLevel
            // 
            this.ProgSetLevel.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.ProgSetLevel.Items.AddRange(new object[] {
            "Brick",
            "Volume",
            "Scanning (Calibrated marks)",
            "Scanning (Template marks)",
            "Lowest"});
            this.ProgSetLevel.Location = new System.Drawing.Point(523, 375);
            this.ProgSetLevel.Name = "ProgSetLevel";
            this.ProgSetLevel.Size = new System.Drawing.Size(280, 21);
            this.ProgSetLevel.TabIndex = 5;
            // 
            // ProgSetSettings
            // 
            this.ProgSetSettings.Location = new System.Drawing.Point(520, 80);
            this.ProgSetSettings.Multiline = true;
            this.ProgSetSettings.Name = "ProgSetSettings";
            this.ProgSetSettings.ScrollBars = System.Windows.Forms.ScrollBars.Both;
            this.ProgSetSettings.Size = new System.Drawing.Size(282, 231);
            this.ProgSetSettings.TabIndex = 4;
            this.ProgSetSettings.WordWrap = false;
            // 
            // ProgSetAuthor
            // 
            this.ProgSetAuthor.Location = new System.Drawing.Point(520, 8);
            this.ProgSetAuthor.Name = "ProgSetAuthor";
            this.ProgSetAuthor.ReadOnly = true;
            this.ProgSetAuthor.Size = new System.Drawing.Size(282, 20);
            this.ProgSetAuthor.TabIndex = 3;
            // 
            // ProgSetDel
            // 
            this.ProgSetDel.Location = new System.Drawing.Point(54, 372);
            this.ProgSetDel.Name = "ProgSetDel";
            this.ProgSetDel.Size = new System.Drawing.Size(40, 24);
            this.ProgSetDel.TabIndex = 2;
            this.ProgSetDel.Text = "Del";
            this.ProgSetDel.Click += new System.EventHandler(this.ProgSetDel_Click);
            // 
            // ProgSetAdd
            // 
            this.ProgSetAdd.Location = new System.Drawing.Point(8, 372);
            this.ProgSetAdd.Name = "ProgSetAdd";
            this.ProgSetAdd.Size = new System.Drawing.Size(40, 24);
            this.ProgSetAdd.TabIndex = 1;
            this.ProgSetAdd.Text = "Add";
            this.ProgSetAdd.Click += new System.EventHandler(this.ProgSetAdd_Click);
            // 
            // ProgSetList
            // 
            this.ProgSetList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader38,
            this.columnHeader39,
            this.columnHeader40,
            this.columnHeader48,
            this.columnHeader49});
            this.ProgSetList.FullRowSelect = true;
            this.ProgSetList.GridLines = true;
            this.ProgSetList.HideSelection = false;
            this.ProgSetList.Location = new System.Drawing.Point(8, 8);
            this.ProgSetList.Name = "ProgSetList";
            this.ProgSetList.Size = new System.Drawing.Size(504, 358);
            this.ProgSetList.TabIndex = 0;
            this.ProgSetList.UseCompatibleStateImageBehavior = false;
            this.ProgSetList.View = System.Windows.Forms.View.Details;
            this.ProgSetList.SelectedIndexChanged += new System.EventHandler(this.OnProgListSelChanged);
            // 
            // columnHeader38
            // 
            this.columnHeader38.Text = "Description";
            this.columnHeader38.Width = 166;
            // 
            // columnHeader39
            // 
            this.columnHeader39.Text = "Executable";
            this.columnHeader39.Width = 99;
            // 
            // columnHeader40
            // 
            this.columnHeader40.Text = "Level";
            this.columnHeader40.Width = 73;
            // 
            // columnHeader48
            // 
            this.columnHeader48.Text = "ID";
            this.columnHeader48.Width = 133;
            // 
            // columnHeader7
            // 
            this.columnHeader7.Text = "Name";
            // 
            // columnHeader8
            // 
            this.columnHeader8.Text = "Value";
            // 
            // columnHeader9
            // 
            this.columnHeader9.Text = "Name";
            // 
            // columnHeader10
            // 
            this.columnHeader10.Text = "Longitude";
            // 
            // columnHeader11
            // 
            this.columnHeader11.Text = "Latitude";
            // 
            // columnHeader12
            // 
            this.columnHeader12.Text = "Local Time Fuse";
            // 
            // columnHeader13
            // 
            this.columnHeader13.Text = "Name";
            // 
            // columnHeader14
            // 
            this.columnHeader14.Text = "Value";
            // 
            // columnHeader15
            // 
            this.columnHeader15.Text = "Name";
            // 
            // columnHeader16
            // 
            this.columnHeader16.Text = "Longitude";
            // 
            // columnHeader17
            // 
            this.columnHeader17.Text = "Latitude";
            // 
            // columnHeader18
            // 
            this.columnHeader18.Text = "Local Time Fuse";
            // 
            // columnHeader19
            // 
            this.columnHeader19.Text = "Name";
            // 
            // columnHeader20
            // 
            this.columnHeader20.Text = "Value";
            // 
            // columnHeader21
            // 
            this.columnHeader21.Text = "Name";
            // 
            // columnHeader22
            // 
            this.columnHeader22.Text = "Longitude";
            // 
            // columnHeader23
            // 
            this.columnHeader23.Text = "Latitude";
            // 
            // columnHeader24
            // 
            this.columnHeader24.Text = "Local Time Fuse";
            // 
            // columnHeader25
            // 
            this.columnHeader25.Text = "Name";
            // 
            // columnHeader26
            // 
            this.columnHeader26.Text = "Longitude";
            // 
            // columnHeader27
            // 
            this.columnHeader27.Text = "Latitude";
            // 
            // columnHeader28
            // 
            this.columnHeader28.Text = "Local Time Fuse";
            // 
            // columnHeader29
            // 
            this.columnHeader29.Text = "Name";
            // 
            // columnHeader30
            // 
            this.columnHeader30.Text = "Longitude";
            // 
            // columnHeader31
            // 
            this.columnHeader31.Text = "Latitude";
            // 
            // columnHeader32
            // 
            this.columnHeader32.Text = "Local Time Fuse";
            // 
            // columnHeader49
            // 
            this.columnHeader49.Text = "Mark Sets";
            // 
            // label4
            // 
            this.label4.Location = new System.Drawing.Point(518, 317);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(76, 24);
            this.label4.TabIndex = 11;
            this.label4.Text = "Mark Sets";
            this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // ProgSetCheckList
            // 
            this.ProgSetCheckList.FormattingEnabled = true;
            this.ProgSetCheckList.Items.AddRange(new object[] {
            "Spot Optical",
            "Lateral X-ray",
            "Spot X-ray"});
            this.ProgSetCheckList.Location = new System.Drawing.Point(608, 317);
            this.ProgSetCheckList.Name = "ProgSetCheckList";
            this.ProgSetCheckList.Size = new System.Drawing.Size(195, 49);
            this.ProgSetCheckList.TabIndex = 12;
            this.ProgSetCheckList.ThreeDCheckBoxes = true;
            // 
            // MainForm
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(826, 438);
            this.Controls.Add(this.GroupTab);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Name = "MainForm";
            this.Text = "Easy Site Management";
            this.Load += new System.EventHandler(this.OnLoad);
            this.GroupTab.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage1.PerformLayout();
            this.tabPage2.ResumeLayout(false);
            this.tabPage2.PerformLayout();
            this.tabPage3.ResumeLayout(false);
            this.tabPage3.PerformLayout();
            this.tabPage4.ResumeLayout(false);
            this.tabPage4.PerformLayout();
            this.ResumeLayout(false);

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

		private SySal.OperaDb.OperaDbCredentials Cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();

		private long UserId;

		private void OnLoad(object sender, System.EventArgs e)
		{
			try
			{
				Conn = new SySal.OperaDb.OperaDbConnection(Cred.DBServer, Cred.DBUserName, Cred.DBPassword);
				Conn.Open();
				UserId = SySal.OperaDb.ComputingInfrastructure.User.CheckLogin(Cred.OPERAUserName, Cred.OPERAPassword, Conn, null);
				if (UserId < 0) throw new Exception("User cannot log in as an administrator");
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Startup error.");
				Close();
				return;
			}
			try
			{
				if (SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM ALL_TABLES WHERE OWNER = 'OPERA' AND TABLE_NAME = 'LZ_SITEVARS'", Conn, null).ExecuteScalar()) != 1 ||
					SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM ALL_TABLES WHERE OWNER = 'OPERA' AND TABLE_NAME = 'LZ_MACHINEVARS'", Conn, null).ExecuteScalar()) != 1)
					new SySal.OperaDb.OperaDbCommand(
						"BEGIN " +
						"EXECUTE IMMEDIATE 'CREATE TABLE OPERA.LZ_SITEVARS (NAME varchar2(255) not null, VALUE varchar2(255) not null, constraint PK_LZ_SITEVARS PRIMARY KEY (NAME)) TABLESPACE OPERASYSTEM'; " +
						"EXECUTE IMMEDIATE 'CREATE TABLE OPERA.LZ_MACHINEVARS (ID_MACHINE NUMBER(*,0) not null, NAME varchar2(255) not null, VALUE varchar2(255) not null, constraint PK_LZ_MACHINES PRIMARY KEY (ID_MACHINE, NAME), constraint FK_MACHINES_LZTB FOREIGN KEY(ID_MACHINE) REFERENCES OPERA.TB_MACHINES(ID)) TABLESPACE OPERASYSTEM'; " +
						"EXECUTE IMMEDIATE 'CREATE PUBLIC SYNONYM LZ_SITEVARS FOR OPERA.LZ_SITEVARS'; " +
						"EXECUTE IMMEDIATE 'CREATE PUBLIC SYNONYM LZ_MACHINEVARS FOR OPERA.LZ_MACHINEVARS'; " +
						"END;", 					
						Conn, null).ExecuteNonQuery();
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Site definition tables already exist.");
			}
			SiteList.Columns[0].Width = SiteList.Width / 4;
			SiteList.Columns[1].Width = SiteList.Width / 4;
			SiteList.Columns[2].Width = SiteList.Width / 4;
			SiteList.Columns[3].Width = SiteList.Width / 4;
			SiteVars.Columns[0].Width = SiteVars.Width / 2;
			SiteVars.Columns[1].Width = SiteVars.Width / 2;
			MachineList.Columns[0].Width = MachineList.Width / 3;
			MachineList.Columns[1].Width = MachineList.Width / 3;
			MachineList.Columns[2].Width = MachineList.Width / 3;
			MachineVars.Columns[0].Width = MachineVars.Width / 2;
			MachineVars.Columns[1].Width = MachineVars.Width / 2;
			ProgSetList.Columns[0].Width = ProgSetList.Width / 4;			
			ProgSetList.Columns[1].Width = ProgSetList.Width / 4;
			ProgSetList.Columns[2].Width = ProgSetList.Width / 4;
			ProgSetList.Columns[3].Width = ProgSetList.Width / 4;
			UserList.Columns[0].Width = UserList.Width / 7;
			UserList.Columns[1].Width = UserList.Width / 7;
			UserList.Columns[2].Width = UserList.Width / 7;
			UserList.Columns[3].Width = UserList.Width / 7;
			UserList.Columns[4].Width = UserList.Width / 7;
			UserList.Columns[5].Width = UserList.Width / 7;
			UserList.Columns[6].Width = UserList.Width / 7;
			try
			{
				System.Data.DataSet ds = new System.Data.DataSet();
				new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, NAME, LONGITUDE, LATITUDE, LOCALTIMEFUSE FROM TB_SITES", Conn, null).Fill(ds);
				foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
				{
					ListViewItem lvi = SiteList.Items.Add(dr[1].ToString());
					lvi.SubItems.Add(dr[2].ToString());
					lvi.SubItems.Add(dr[3].ToString());
					lvi.SubItems.Add(dr[4].ToString());
					lvi.Tag = SySal.OperaDb.Convert.ToInt64(dr[0]);
				}
				long idsite = 0;				
				try
				{
					idsite = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME = 'ID_SITE'", Conn, null).ExecuteScalar());
				}
				catch (Exception) {}
				foreach (ListViewItem lvs in SiteList.Items)
					if (Convert.ToInt64(lvs.Tag) == idsite)
						SetSite(idsite);
				ShowProgramSettings();
				ShowUsers();
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error initializing sites", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
		}

		private void EnsurePresent(string name, string defaultvalue, SySal.OperaDb.OperaDbTransaction trans)
		{
			if (Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM LZ_SITEVARS WHERE NAME = '" + name + "'", Conn, trans).ExecuteScalar()) == 0)
				new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_SITEVARS (NAME, VALUE) VALUES ('" + name + "', '" + defaultvalue +"')", Conn, trans).ExecuteNonQuery();
		}

		private void SiteMarkCurrent_Click(object sender, System.EventArgs e)
		{
			SySal.OperaDb.OperaDbTransaction trans = null;
			if (SiteList.SelectedItems.Count != 1) return;
			try
			{
				trans = Conn.BeginTransaction();
				new SySal.OperaDb.OperaDbCommand("DELETE FROM LZ_SITEVARS WHERE NAME = 'ID_SITE'", Conn, trans).ExecuteNonQuery();
				new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_SITEVARS (NAME, VALUE) VALUES ('ID_SITE', '" + SiteList.SelectedItems[0].Tag.ToString() + "')", Conn, trans).ExecuteNonQuery();
				EnsurePresent("ExeRepository", "c:\\", trans);
				EnsurePresent("ScratchDir", "c:\\", trans);
				EnsurePresent("DriverDir", "c:\\", trans);
				EnsurePresent("RawDataDir", "c:\\", trans);
				EnsurePresent("TaskDir", "c:\\", trans);
				EnsurePresent("ArchivedTaskDir", "c:\\", trans);
				EnsurePresent("BM_DataProcSrvMonitorInterval", "60", trans);
				EnsurePresent("BM_ImpersonateBatchUser", "true", trans);
				EnsurePresent("BM_ResultLiveSeconds", "600", trans);
				EnsurePresent("DPS_PeakWorkingSetMB", "128", trans);
				EnsurePresent("DPS_MachinePowerClass", "5", trans);
				EnsurePresent("DPS_LowPriority", "true", trans);
				EnsurePresent("DPS_ResultLiveSeconds", "600", trans);							
				trans.Commit();
				SetSite(Convert.ToInt64(SiteList.SelectedItems[0].Tag));
			}
			catch (Exception x)
			{
				if (trans != null) trans.Rollback();
				MessageBox.Show(x.Message, "Error setting current site", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
		}

		private void SetSite(long id)
		{
			SiteVars.Items.Clear();
			System.Data.DataSet ds = new System.Data.DataSet();
			new SySal.OperaDb.OperaDbDataAdapter("SELECT NAME, VALUE FROM LZ_SITEVARS WHERE NAME <> 'ID_SITE'", Conn, null).Fill(ds);
			foreach (System.Data.DataRow dr in ds.Tables[0].Rows)			
				SiteVars.Items.Add(dr[0].ToString()).SubItems.Add(dr[1].ToString());
			ds = new System.Data.DataSet();
			MachineList.Items.Clear();
			MachineVars.Items.Clear();
			new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, NAME, ADDRESS, ISSCANNINGSERVER, ISDATAPROCESSINGSERVER, ISBATCHSERVER, ISDATABASESERVER, ISWEBSERVER FROM TB_MACHINES WHERE ID_SITE = " + id.ToString(), Conn, null).Fill(ds);
			foreach (System.Data.DataRow dr in ds.Tables[0].Rows)			
			{
				ListViewItem lvi = MachineList.Items.Add(dr[1].ToString());
				lvi.SubItems.Add(dr[2].ToString());
				string f = "";
				if (Convert.ToInt32(dr[3]) == 1) f += "S";
				if (Convert.ToInt32(dr[4]) == 1) f += "P";
				if (Convert.ToInt32(dr[5]) == 1) f += "B";
				if (Convert.ToInt32(dr[6]) == 1) f += "D";
				if (Convert.ToInt32(dr[7]) == 1) f += "W";
				lvi.SubItems.Add(f);
				lvi.Tag = SySal.OperaDb.Convert.ToInt64(dr[0]);
			}
			SetMachine(0);
		}

		private void OnSiteVarSelChanged(object sender, System.EventArgs e)
		{
			if (SiteVars.SelectedItems.Count != 1) return;
			SiteVarName.Text = SiteVars.SelectedItems[0].SubItems[0].Text;
			SiteVarValue.Text = SiteVars.SelectedItems[0].SubItems[1].Text;
		}

		private void SiteVarSet_Click(object sender, System.EventArgs e)
		{
			SiteVarName.Text = SiteVarName.Text.Trim();
			SiteVarValue.Text = SiteVarValue.Text.Trim();
			if (SiteVarName.Text.Length == 0) return;
			SySal.OperaDb.OperaDbTransaction trans = null;
			try
			{
				trans = Conn.BeginTransaction();
				new SySal.OperaDb.OperaDbCommand("DELETE FROM LZ_SITEVARS WHERE NAME = '" + SiteVarName.Text + "'", Conn, trans).ExecuteNonQuery();
                SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_SITEVARS (NAME, VALUE) VALUES (:n, :v)", Conn, trans);
                cmd.Parameters.Add("n", SySal.OperaDb.OperaDbType.String, ParameterDirection.Input).Value = SiteVarName.Text;
                cmd.Parameters.Add("v", SySal.OperaDb.OperaDbType.String, ParameterDirection.Input).Value = SiteVarValue.Text;
                cmd.ExecuteNonQuery();
				trans.Commit();
				SetSite(SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME = 'ID_SITE'", Conn).ExecuteScalar()));
			}
			catch (Exception x)
			{
				if (trans != null) trans.Rollback();
				MessageBox.Show(x.Message, "Error setting variable", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}		
		}

		private void SiteVarDel_Click(object sender, System.EventArgs e)
		{
			if (SiteVars.SelectedItems.Count != 1) return;
			try
			{
				new SySal.OperaDb.OperaDbCommand("DELETE FROM LZ_SITEVARS WHERE NAME = '" + SiteVars.SelectedItems[0].SubItems[0].Text + "'", Conn, null).ExecuteNonQuery();
				SetSite(SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME = 'ID_SITE'", Conn).ExecuteScalar()));
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error deleting variable", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}				
		}

		private void MachineAdd_Click(object sender, System.EventArgs e)
		{
			MachineName.Text = MachineName.Text.Trim();
			MachineAddr.Text = MachineAddr.Text.Trim();
			MachineFunc.Text = MachineFunc.Text.ToUpper();
			try
			{
				long idsite = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME = 'ID_SITE'", Conn).ExecuteScalar());
				SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_MACHINE(" + idsite + ", '" + MachineName.Text + "','" + MachineAddr.Text + "'," + 
					((MachineFunc.Text.IndexOf("S") >= 0) ? "1" : "0") + "," +
					((MachineFunc.Text.IndexOf("B") >= 0) ? "1" : "0") + "," +
					((MachineFunc.Text.IndexOf("P") >= 0) ? "1" : "0") + "," +
					((MachineFunc.Text.IndexOf("W") >= 0) ? "1" : "0") + "," +
					((MachineFunc.Text.IndexOf("D") >= 0) ? "1" : "0") + ", :newid)", Conn, null);
				cmd.Parameters.Add("newid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				cmd.ExecuteNonQuery();
				SetSite(idsite);
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error adding machine", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}										
		}

		private void MachineSet_Click(object sender, System.EventArgs e)
		{
			MachineName.Text = MachineName.Text.Trim();
			MachineAddr.Text = MachineAddr.Text.Trim();
			MachineFunc.Text = MachineFunc.Text.ToUpper();
			if (MachineList.SelectedItems.Count != 1) return;
			try
			{
				SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("CALL PC_SET_MACHINE(" + MachineList.SelectedItems[0].Tag.ToString() + ", '" + MachineName.Text + "','" + MachineAddr.Text + "'," + 
					((MachineFunc.Text.IndexOf("S") >= 0) ? "1" : "0") + "," +
					((MachineFunc.Text.IndexOf("B") >= 0) ? "1" : "0") + "," +
					((MachineFunc.Text.IndexOf("P") >= 0) ? "1" : "0") + "," +
					((MachineFunc.Text.IndexOf("W") >= 0) ? "1" : "0") + "," +
					((MachineFunc.Text.IndexOf("D") >= 0) ? "1" : "0") + ")", Conn, null);
				cmd.Parameters.Add("newid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				cmd.ExecuteNonQuery();
				SetSite(SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME = 'ID_SITE'", Conn).ExecuteScalar()));
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error setting machine", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}				
		}

		private void MachineDel_Click(object sender, System.EventArgs e)
		{
			if (MachineList.SelectedItems.Count != 1) return;
			try
			{
				new SySal.OperaDb.OperaDbCommand("CALL PC_DEL_MACHINE(" + MachineList.SelectedItems[0].Tag.ToString() + ")", Conn, null).ExecuteNonQuery();
				SetSite(SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME = 'ID_SITE'", Conn).ExecuteScalar()));
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error deleting machine", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}				
		}

		private void MachineVarSet_Click(object sender, System.EventArgs e)
		{
			MachineVarName.Text = MachineVarName.Text.Trim();
			MachineVarValue.Text = MachineVarValue.Text.Trim();
			if (MachineList.SelectedItems.Count != 1) return;
			if (MachineVarName.Text.Length == 0) return;
			SySal.OperaDb.OperaDbTransaction trans = null;
			try
			{
				trans = Conn.BeginTransaction();
				new SySal.OperaDb.OperaDbCommand("DELETE FROM LZ_MACHINEVARS WHERE NAME = '" + MachineVarName.Text + "' AND ID_MACHINE = " + MachineList.SelectedItems[0].Tag.ToString(), Conn, trans).ExecuteNonQuery();
				SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("INSERT INTO LZ_MACHINEVARS (ID_MACHINE, NAME, VALUE) VALUES (" + MachineList.SelectedItems[0].Tag.ToString() + ",:n,:v)", Conn, trans);
                cmd.Parameters.Add("n", SySal.OperaDb.OperaDbType.String, ParameterDirection.Input).Value = MachineVarName.Text;
                cmd.Parameters.Add("v", SySal.OperaDb.OperaDbType.String, ParameterDirection.Input).Value = MachineVarValue.Text;
                cmd.ExecuteNonQuery();
				trans.Commit();
				SetMachine(Convert.ToInt64(MachineList.SelectedItems[0].Tag));
			}
			catch (Exception x)
			{
				if (trans != null) trans.Rollback();
				MessageBox.Show(x.Message, "Error setting variable", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}				
		}

		private void MachineVarDel_Click(object sender, System.EventArgs e)
		{
			if (MachineVars.SelectedItems.Count != 1) return;
			if (MachineList.SelectedItems.Count != 1) return;
			try
			{
				new SySal.OperaDb.OperaDbCommand("DELETE FROM LZ_MACHINEVARS WHERE NAME = '" + MachineVars.SelectedItems[0].SubItems[0].Text + "' AND ID_MACHINE = " + MachineList.SelectedItems[0].Tag.ToString(), Conn, null).ExecuteNonQuery();
				SetMachine(SySal.OperaDb.Convert.ToInt64(MachineList.SelectedItems[0].Tag));
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error deleting variable", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}						
		}

		private void OnMachineSelChanged(object sender, System.EventArgs e)
		{
			if (MachineList.SelectedItems.Count != 1) return;
			MachineName.Text = MachineList.SelectedItems[0].SubItems[0].Text;
			MachineAddr.Text = MachineList.SelectedItems[0].SubItems[1].Text;
			MachineFunc.Text = MachineList.SelectedItems[0].SubItems[2].Text;
			SetMachine(Convert.ToInt64(MachineList.SelectedItems[0].Tag));
		}

		private void OnMachineVarSelChanged(object sender, System.EventArgs e)
		{
			if (MachineVars.SelectedItems.Count != 1) return;
			MachineVarName.Text = MachineVars.SelectedItems[0].SubItems[0].Text;
			MachineVarValue.Text = MachineVars.SelectedItems[0].SubItems[1].Text;		
		}

		private void SetMachine(long id)
		{
			MachineVars.Items.Clear();
			System.Data.DataSet ds = new System.Data.DataSet();
			new SySal.OperaDb.OperaDbDataAdapter("SELECT NAME, VALUE FROM LZ_MACHINEVARS WHERE ID_MACHINE = " + id, Conn, null).Fill(ds);
			foreach (System.Data.DataRow dr in ds.Tables[0].Rows)			
				MachineVars.Items.Add(dr[0].ToString()).SubItems.Add(dr[1].ToString());
		}

		private void ShowProgramSettings()
		{
			ProgSetList.Items.Clear();
			System.Data.DataSet ds = new System.Data.DataSet();
			new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, DESCRIPTION, EXECUTABLE, DRIVERLEVEL, TEMPLATEMARKS, MARKSET FROM TB_PROGRAMSETTINGS ORDER BY DESCRIPTION ASC", Conn, null).Fill(ds);
			foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
			{
                try
                {
                    ListViewItem lvi = ProgSetList.Items.Add(dr[1].ToString());
                    lvi.SubItems.Add(dr[2].ToString());
                    switch (Convert.ToInt32(dr[3]))
                    {
                        case 0: lvi.SubItems.Add("Lowest"); break;
                        case 1: if (dr[4] == System.DBNull.Value) lvi.SubItems.Add("WARNING - Missing TEMPLATEMARKS");
                            else if (SySal.OperaDb.Convert.ToInt32(dr[4]) == 0) lvi.SubItems.Add("Scanning (Calibrated marks)"); else lvi.SubItems.Add("Scanning (Template marks)"); break;
                        case 2: lvi.SubItems.Add("Volume"); break;
                        case 3: lvi.SubItems.Add("Brick"); break;
                        default: lvi.SubItems.Add("Higher (" + Convert.ToInt32(dr[3]) + ")"); break;
                    }
                    lvi.SubItems.Add(dr[0].ToString());
                    lvi.SubItems.Add(dr[5].ToString());
                    lvi.Tag = SySal.OperaDb.Convert.ToInt64(dr[0]);
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "Data error");
                }
			}
		}

		private void OnProgListSelChanged(object sender, System.EventArgs e)
		{
			if (ProgSetList.SelectedItems.Count == 0) return;
			System.Data.DataSet ds = new System.Data.DataSet();
			new SySal.OperaDb.OperaDbDataAdapter("SELECT NAME, SURNAME FROM VW_USERS INNER JOIN TB_PROGRAMSETTINGS ON (TB_PROGRAMSETTINGS.ID_AUTHOR = VW_USERS.ID) WHERE TB_PROGRAMSETTINGS.ID = " + ProgSetList.SelectedItems[0].Tag.ToString(), Conn, null).Fill(ds);
			ProgSetAuthor.Text = ds.Tables[0].Rows[0][0].ToString() + " " + ds.Tables[0].Rows[0][1].ToString();
			ProgSetSettings.Text = new SySal.OperaDb.OperaDbCommand("SELECT SETTINGS FROM TB_PROGRAMSETTINGS WHERE ID = " + ProgSetList.SelectedItems[0].Tag.ToString(), Conn, null).ExecuteScalar().ToString();
			ProgSetExecutable.Text = ProgSetList.SelectedItems[0].SubItems[1].Text;			
			ProgSetDescription.Text = ProgSetList.SelectedItems[0].SubItems[0].Text;
			ProgSetLevel.Text = ProgSetList.SelectedItems[0].SubItems[2].Text;
            string marksets = "";
            try
            {
                marksets = ProgSetList.SelectedItems[0].SubItems[4].Text;
            }
            catch (Exception) { };
            ProgSetCheckList.SetItemChecked(0, marksets.IndexOf(SySal.DAQSystem.Drivers.MarkChar.SpotOptical) >= 0);
            ProgSetCheckList.SetItemChecked(1, marksets.IndexOf(SySal.DAQSystem.Drivers.MarkChar.LineXRay) >= 0);
            ProgSetCheckList.SetItemChecked(2, marksets.IndexOf(SySal.DAQSystem.Drivers.MarkChar.SpotXRay) >= 0);
		}

		private void ProgSetAdd_Click(object sender, System.EventArgs e)
		{
			try
			{
				long idsite = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME = 'ID_SITE'", Conn).ExecuteScalar());
				SySal.OperaDb.ComputingInfrastructure.UserPermission [] perm = new SySal.OperaDb.ComputingInfrastructure.UserPermission[1];
				perm[0].DB_Site_Id = idsite;
				perm[0].Designator = SySal.OperaDb.ComputingInfrastructure.UserPermissionDesignator.Administer;
				perm[0].Value = SySal.OperaDb.ComputingInfrastructure.UserPermissionTriState.Grant;
				if (SySal.OperaDb.ComputingInfrastructure.User.CheckAccess(UserId, perm, true, Conn, null) == false) throw new Exception("User is not an administrator");
				int driverlevel, templatemarks = 0;
				if (String.Compare(ProgSetLevel.SelectedItem.ToString(), "Lowest", true) == 0) driverlevel = (int)SySal.DAQSystem.Drivers.DriverType.Lowest;
				else if (String.Compare(ProgSetLevel.SelectedItem.ToString(), "Scanning (Calibrated marks)", true) == 0) driverlevel = (int)SySal.DAQSystem.Drivers.DriverType.Scanning;
				else if (String.Compare(ProgSetLevel.SelectedItem.ToString(), "Scanning (Template marks)", true) == 0) 
				{
					driverlevel = (int)SySal.DAQSystem.Drivers.DriverType.Scanning;
					templatemarks = 1;
				}
				else if (String.Compare(ProgSetLevel.SelectedItem.ToString(), "Volume", true) == 0) driverlevel = (int)SySal.DAQSystem.Drivers.DriverType.Volume;
				else if (String.Compare(ProgSetLevel.SelectedItem.ToString(), "Brick", true) == 0) driverlevel = (int)SySal.DAQSystem.Drivers.DriverType.Brick;
				else throw new Exception("Unknown driver level");
                string markset = "";
                if (ProgSetCheckList.GetItemChecked(0)) markset += SySal.DAQSystem.Drivers.MarkChar.SpotOptical;
                if (ProgSetCheckList.GetItemChecked(1)) markset += SySal.DAQSystem.Drivers.MarkChar.LineXRay;
                if (ProgSetCheckList.GetItemChecked(2)) markset += SySal.DAQSystem.Drivers.MarkChar.SpotXRay;
                if (markset.Length > 0) markset = "'" + markset + "'";
                else markset = "NULL";
				SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_PROGRAMSETTINGS('" + ProgSetDescription.Text + "','" + ProgSetExecutable.Text + "'," + UserId + "," + driverlevel + "," + ((driverlevel == 1) ? templatemarks.ToString() : "NULL") + ", " + markset + ", :settings ,:newid)", Conn, null);
				cmd.Parameters.Add("settings", SySal.OperaDb.OperaDbType.CLOB, System.Data.ParameterDirection.Input).Value = ProgSetSettings.Text;
				cmd.Parameters.Add("newid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				cmd.ExecuteNonQuery();
				ShowProgramSettings();
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error adding program", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}						
		}

		private void ProgSetDel_Click(object sender, System.EventArgs e)
		{
			if (ProgSetList.SelectedItems.Count != 1) return;
			try
			{
				new SySal.OperaDb.OperaDbCommand("CALL PC_DEL_PROGRAMSETTINGS(" + ProgSetList.SelectedItems[0].Tag.ToString() + ")", Conn, null).ExecuteNonQuery();
				ShowProgramSettings();
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error deleting program settings", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}						
		}

		private void ShowUsers()
		{
			UserList.Items.Clear();
			System.Data.DataSet ds = new System.Data.DataSet();
			new SySal.OperaDb.OperaDbDataAdapter("SELECT ID, USERNAME, NAME, SURNAME, INSTITUTION, EMAIL, ADDRESS, PHONE FROM VW_USERS ORDER BY USERNAME ASC", Conn, null).Fill(ds);
			foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
			{
				ListViewItem lvi = UserList.Items.Add(dr[1].ToString());
				lvi.SubItems.Add(dr[2].ToString());
				lvi.SubItems.Add(dr[3].ToString());
				lvi.SubItems.Add(dr[4].ToString());
				lvi.SubItems.Add(dr[5].ToString());
				lvi.SubItems.Add(dr[6].ToString());
				lvi.SubItems.Add(dr[7].ToString());
				lvi.Tag = SySal.OperaDb.Convert.ToInt64(dr[0]);
			}
		}

		private void UserAdd_Click(object sender, System.EventArgs e)
		{
			SySal.OperaDb.OperaDbTransaction trans = null;
			try
			{				
				long idsite = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME = 'ID_SITE'", Conn).ExecuteScalar());
				SySal.OperaDb.ComputingInfrastructure.UserPermission [] perm = new SySal.OperaDb.ComputingInfrastructure.UserPermission[1];
				perm[0].DB_Site_Id = idsite;
				perm[0].Designator = SySal.OperaDb.ComputingInfrastructure.UserPermissionDesignator.Administer;
				perm[0].Value = SySal.OperaDb.ComputingInfrastructure.UserPermissionTriState.Grant;
				if (SySal.OperaDb.ComputingInfrastructure.User.CheckAccess(UserId, perm, true, Conn, null) == false) throw new Exception("User is not an administrator");

				trans = Conn.BeginTransaction();
				SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_USER(" + idsite.ToString() + ",'" + UserUsername.Text + "','" + UserPassword.Text + "','" + UserName.Text + "','" + UserSurname.Text + "','" + UserInstitution.Text + "','" + UserEmail.Text +"','" + UserAddress.Text + "','" + UserPhone.Text + "',:newid)", Conn, trans);
				cmd.Parameters.Add("newid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
				cmd.ExecuteNonQuery();
				long newid = SySal.OperaDb.Convert.ToInt64(cmd.Parameters[0].Value);
				UserPermissions.Text = UserPermissions.Text.ToUpper();
				new SySal.OperaDb.OperaDbCommand("CALL PC_SET_PRIVILEGES(" + newid + "," + idsite + "," +
					((UserPermissions.Text.IndexOf("S") >= 0) ? "1" : "0") + ", " +
					((UserPermissions.Text.IndexOf("W") >= 0) ? "1" : "0") + ", " +
					((UserPermissions.Text.IndexOf("P") >= 0) ? "1" : "0") + ", " +
					((UserPermissions.Text.IndexOf("D") >= 0) ? "1" : "0") + ", " +
					((UserPermissions.Text.IndexOf("B") >= 0) ? "1" : "0") + ", " +
					((UserPermissions.Text.IndexOf("A") >= 0) ? "1" : "0") + ")", Conn, trans).ExecuteNonQuery();				
				trans.Commit();
				ShowUsers();
			}
			catch (Exception x)
			{
				if (trans != null) trans.Rollback();
				MessageBox.Show(x.Message, "Error adding user info", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}		
		}

		private void UserSet_Click(object sender, System.EventArgs e)
		{
			SySal.OperaDb.OperaDbTransaction trans = null;
			try
			{				
				long idsite = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME = 'ID_SITE'", Conn).ExecuteScalar());
				SySal.OperaDb.ComputingInfrastructure.UserPermission [] perm = new SySal.OperaDb.ComputingInfrastructure.UserPermission[1];
				perm[0].DB_Site_Id = idsite;
				perm[0].Designator = SySal.OperaDb.ComputingInfrastructure.UserPermissionDesignator.Administer;
				perm[0].Value = SySal.OperaDb.ComputingInfrastructure.UserPermissionTriState.Grant;
				if (SySal.OperaDb.ComputingInfrastructure.User.CheckAccess(UserId, perm, true, Conn, null) == false) throw new Exception("User is not an administrator");

				if (UserList.SelectedItems.Count != 1) return;
				trans = Conn.BeginTransaction();
				new SySal.OperaDb.OperaDbCommand("CALL PC_SET_USER(" + UserList.SelectedItems[0].Tag.ToString() + ", '" + UserUsername.Text + "','" + UserPassword.Text + "','" + UserName.Text + "','" + UserSurname.Text + "','" + UserInstitution.Text + "','" + UserEmail.Text +"','" + UserAddress.Text + "','" + UserPhone.Text + "')", Conn, trans).ExecuteNonQuery();
				UserPermissions.Text = UserPermissions.Text.ToUpper();
				new SySal.OperaDb.OperaDbCommand("CALL PC_SET_PRIVILEGES(" + UserList.SelectedItems[0].Tag.ToString() + "," + idsite + "," +
					((UserPermissions.Text.IndexOf("S") >= 0) ? "1" : "0") + ", " +
					((UserPermissions.Text.IndexOf("W") >= 0) ? "1" : "0") + ", " +
					((UserPermissions.Text.IndexOf("P") >= 0) ? "1" : "0") + ", " +
					((UserPermissions.Text.IndexOf("D") >= 0) ? "1" : "0") + ", " +
					((UserPermissions.Text.IndexOf("B") >= 0) ? "1" : "0") + ", " +
					((UserPermissions.Text.IndexOf("A") >= 0) ? "1" : "0") + ")", Conn, trans).ExecuteNonQuery();				
				trans.Commit();
				ShowUsers();
			}
			catch (Exception x)
			{
				if (trans != null) trans.Rollback();
				MessageBox.Show(x.Message, "Error setting user info", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}				
		}

		private void UserDel_Click(object sender, System.EventArgs e)
		{
			try
			{				
				long idsite = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME = 'ID_SITE'", Conn).ExecuteScalar());
				SySal.OperaDb.ComputingInfrastructure.UserPermission [] perm = new SySal.OperaDb.ComputingInfrastructure.UserPermission[1];
				perm[0].DB_Site_Id = idsite;
				perm[0].Designator = SySal.OperaDb.ComputingInfrastructure.UserPermissionDesignator.Administer;
				perm[0].Value = SySal.OperaDb.ComputingInfrastructure.UserPermissionTriState.Grant;
				if (SySal.OperaDb.ComputingInfrastructure.User.CheckAccess(UserId, perm, true, Conn, null) == false) throw new Exception("User is not an administrator");

				if (UserList.SelectedItems.Count != 1) return;
				new SySal.OperaDb.OperaDbCommand("CALL PC_DEL_USER(" + UserList.SelectedItems[0].Tag.ToString() + ")", Conn, null).ExecuteNonQuery();
				ShowUsers();
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error reading user info", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}			
		}

		private void OnUserSelChanged(object sender, System.EventArgs e)
		{
			try
			{
				long idsite = SySal.OperaDb.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME = 'ID_SITE'", Conn).ExecuteScalar());
				SySal.OperaDb.ComputingInfrastructure.UserPermission [] perm = new SySal.OperaDb.ComputingInfrastructure.UserPermission[1];
				perm[0].DB_Site_Id = idsite;
				perm[0].Designator = SySal.OperaDb.ComputingInfrastructure.UserPermissionDesignator.Administer;
				perm[0].Value = SySal.OperaDb.ComputingInfrastructure.UserPermissionTriState.Grant;
				if (SySal.OperaDb.ComputingInfrastructure.User.CheckAccess(UserId, perm, true, Conn, null) == false) throw new Exception("User is not an administrator");

				if (UserList.SelectedItems.Count != 1) return;
				UserUsername.Text = UserList.SelectedItems[0].SubItems[0].Text;			
				UserName.Text = UserList.SelectedItems[0].SubItems[1].Text;
				UserSurname.Text = UserList.SelectedItems[0].SubItems[2].Text;
				UserInstitution.Text = UserList.SelectedItems[0].SubItems[3].Text;
				UserEmail.Text = UserList.SelectedItems[0].SubItems[4].Text;
				UserAddress.Text = UserList.SelectedItems[0].SubItems[5].Text;
				UserPhone.Text = UserList.SelectedItems[0].SubItems[6].Text;

				UserPassword.Text = new SySal.OperaDb.OperaDbCommand("SELECT PWD FROM TB_USERS WHERE ID = " + UserList.SelectedItems[0].Tag.ToString(), Conn, null).ExecuteScalar().ToString();
				string p = "";
				SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("CALL PC_GET_PRIVILEGES(" + UserList.SelectedItems[0].Tag.ToString() + ", " +
					new SySal.OperaDb.OperaDbCommand("SELECT VALUE FROM LZ_SITEVARS WHERE NAME = 'ID_SITE'", Conn).ExecuteScalar().ToString() + ",'" 
					+ UserPassword.Text + "',:p_scan,:p_weban,:p_dataproc,:p_datadwnl,:p_procstart,:p_admin)", Conn, null);
				cmd.Parameters.Add("p_scan", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Output);
				cmd.Parameters.Add("p_weban", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Output);
				cmd.Parameters.Add("p_dataproc", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Output);
				cmd.Parameters.Add("p_datadwnl", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Output);
				cmd.Parameters.Add("p_procstart", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Output);
				cmd.Parameters.Add("p_admin", SySal.OperaDb.OperaDbType.Int, System.Data.ParameterDirection.Output);
				cmd.ExecuteNonQuery();
				if (SySal.OperaDb.Convert.ToInt32(cmd.Parameters[0].Value) == 1) p += "S";
				if (SySal.OperaDb.Convert.ToInt32(cmd.Parameters[1].Value) == 1) p += "W";
				if (SySal.OperaDb.Convert.ToInt32(cmd.Parameters[2].Value) == 1) p += "P";
				if (SySal.OperaDb.Convert.ToInt32(cmd.Parameters[3].Value) == 1) p += "D";
				if (SySal.OperaDb.Convert.ToInt32(cmd.Parameters[4].Value) == 1) p += "B";
				if (SySal.OperaDb.Convert.ToInt32(cmd.Parameters[5].Value) == 1) p += "A";
				UserPermissions.Text = p;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Error reading user info", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}						
		}
	}
}
