using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.NExTScanner
{
    public partial class SySalMainForm : Form, IScannerDataDisplay
    {
        static SySalMainForm s_TheMainForm = null;

        internal static SySalMainForm TheMainForm { get { return s_TheMainForm; } }

        public SySalMainForm()
        {
            InitializeComponent();
            MainMenu = new MenuItem[]
            {
                new MenuItem("Setup", 
                    new MenuItem[]
                    {
                        new MenuItem("Scanner", new MenuItem.dOnMenuItem(this.OnMenu_Setup_Scanner), MenuItem.MenuItemEnablerTrue, "Set up the general properties of the Scanner:\r\nchoose directories and hardware control libraries."),
                        new MenuItem("Stage", new MenuItem.dOnMenuItem(this.OnMenu_Setup_Stage), new MenuItem.dMenuItemEnabler(delegate() { return TheScanner.iStage != null; }), "Configure the chosen stage control library:\r\ndefine encoder/micron and step/micron conversions,\r\nlimiters, speed limits and reference positions."),
                        new MenuItem("Grabber", new MenuItem.dOnMenuItem(this.OnMenu_Setup_Grabber), new MenuItem.dMenuItemEnabler(delegate() { return TheScanner.iGrab != null; }), "Configure the camera, grabbing mode, memory allocation, etc.."),
                        new MenuItem("GPU", new MenuItem.dOnMenuItem(this.OnMenu_Setup_GPU), new MenuItem.dMenuItemEnabler(delegate() { return TheScanner.iGPU != null && TheScanner.iGPU.Length > 0; }), "Choose the GPU devices to be used for image processing and set up their operating parameters."),
                        new MenuItem("Imaging", new MenuItem.dOnMenuItem(this.OnMenu_Setup_Imaging), new MenuItem.dMenuItemEnabler(delegate() { return TheScanner.iGPU != null && TheScanner.iGPU.Length > 0; }), "Define the features of the objective:\r\nmagnification factor and various corrections to be applied."),
                        new MenuItem("NExT", null, MenuItem.MenuItemEnablerFalse, "Configure the Scanner to work in a SySal.NExT environment as a server.")
                    }, new MenuItem.dMenuItemEnabler(delegate() { return TheScanner.iStage != null; }), "Basic setup to start working with the Scanner."
                ),
                new MenuItem("ScanServer", 
                    new MenuItem[]
                    {
                        new MenuItem("Start", new MenuItem.dOnMenuItem(this.OnMenu_StartScanServer), new MenuItem.dMenuItemEnabler(delegate() { return TheScanner.NSS_connected == false; }), "Starts listening."),
                        new MenuItem("Stop", new MenuItem.dOnMenuItem(this.OnMenu_StopScanServer), new MenuItem.dMenuItemEnabler(delegate() { return TheScanner.NSS_connected ; }), "Stop listening\r\n(the current action will be completed).")
                    }, new MenuItem.dMenuItemEnabler(delegate() { return TheScanner.iGPU != null && TheScanner.iGPU.Length > 0 && TheScanner.iStage != null; }), "Have this program listen and execute task request from the network."
                ),
                new MenuItem("Acquisition", 
                    new MenuItem[]
                    {
                        new MenuItem("Marks", new MenuItem.dOnMenuItem(this.OnMenu_AcquireMarks), MenuItem.MenuItemEnablerTrue, "Define the reference marks on the current plate."),
                        new MenuItem("Quasi-static", new MenuItem.dOnMenuItem(this.OnMenu_QuasiStaticAcquisition), MenuItem.MenuItemEnablerTrue, "Quasi-static acquisition, useful for high-quality data taking\r\nand building samples of images to work out optical corrections\r\nor to develop image handling/tracking algorithms."),
                        new MenuItem("Full speed", new MenuItem.dOnMenuItem(this.OnMenu_FullSpeedAcquisition), MenuItem.MenuItemEnablerTrue, "Acquire data at the maximum possible speed with the standard procedure.")
                    }, new MenuItem.dMenuItemEnabler(delegate() { return TheScanner.NSS_connected == false && TheScanner.iGPU != null && TheScanner.iGPU.Length > 0 && TheScanner.iStage != null; }), "Perform user-defined scanning tasks.\r\nThis is primarily a tool to define scanning parameters,\r\nbut some simple scanning tasks may be accomplished too."
                )
            };
            m_UpMenuButton = new SySal.SySalNExTControls.SySalButton();
            m_UpMenuButton.Visible = false;
            m_UpMenuButton.Text = "<<";
            m_UpMenuButton.BackColor = Color.Transparent;
            m_UpMenuButton.AutoSize = true;
            m_UpMenuButton.ForeColor = Color.DodgerBlue;
            m_UpMenuButton.FocusedColor = Color.Navy;
            m_UpMenuButton.Font = new Font("Segoe UI", 14);
            SetMenuUpLinks(MainMenu);
            m_CurrentMenu = MainMenu;
            s_TheMainForm = this;
        }

        SySal.SySalNExTControls.SySalButton m_UpMenuButton;

        void SetMenuUpLinks(MenuItem [] mis)
        {
            foreach (MenuItem mi in mis)            
            {
                if (mi.Action is MenuItem [])
                {
                    foreach (MenuItem mid in (MenuItem [])mi.Action)
                        mid.UpMenu = mis;
                    SetMenuUpLinks((MenuItem [])mi.Action);
                }
            }
        }

        MenuItem[] m_CurrentMenu = null;

        void EnableCurrentMenu()
        {
            foreach (MenuItem mi in m_CurrentMenu)
                mi.Button.Enabled = mi.Enabler();
        }

        void SetCurrentMenu()
        {
            MenuFlowLayout.SuspendLayout();
            foreach (Control ctl in MenuFlowLayout.Controls)
            {
                ctl.Visible = false;
                ctl.Dock = DockStyle.None;
                ctl.Parent = null;                
            }
            MenuFlowLayout.Controls.Clear();
            if (m_CurrentMenu[0].UpMenu != null)
            {
                MenuFlowLayout.Controls.Add(m_UpMenuButton);
                m_UpMenuButton.Parent = MenuFlowLayout;
                m_UpMenuButton.Dock = DockStyle.Left;
                m_UpMenuButton.Click -= OnMenu_Up_Clicked;
                m_UpMenuButton.Click += OnMenu_Up_Clicked;
                m_UpMenuButton.Visible = true;                
            }
            foreach (MenuItem mi in m_CurrentMenu)
            {
                MenuFlowLayout.Controls.Add(mi.Button);
                mi.Button.Parent = MenuFlowLayout;
                mi.Button.Dock = DockStyle.Left;
                mi.Button.Click -= OnMenu_Button_Clicked;
                mi.Button.Click += OnMenu_Button_Clicked;
                mi.Button.Visible = true;
                mi.Button.Enabled = mi.Enabler();
                MainToolTip.SetToolTip(mi.Button, mi.ToolTipText);
            }
            MenuFlowLayout.Invalidate();
            MenuFlowLayout.ResumeLayout();
        }

        private void OnLeave(object sender, EventArgs e)
        {
            if (sender is Button)
            {
                ((Button)sender).ForeColor = Color.DodgerBlue;
                ((Button)sender).Refresh();
            }
        }

        private void OnEnter(object sender, EventArgs e)
        {
            if (sender is Button)
            {
                ((Button)sender).ForeColor = Color.Navy;
                ((Button)sender).Refresh();
            }
        }

        private void OnSetupClicked(object sender, EventArgs e)
        {
            MessageBox.Show("Setup");
        }

        private void OnBackpanelPaint(object sender, PaintEventArgs e)
        {
            e.Graphics.FillRectangle(new System.Drawing.Drawing2D.LinearGradientBrush(this.ClientRectangle, /*Color.Silver*/ Color.LightSteelBlue, Color.White, 90), this.ClientRectangle);
        }

        private void sysBtnClose_Click(object sender, EventArgs e)
        {
            (this.TopLevelControl as Form).Close();
        }

        private void sysBtnMinimize_Click(object sender, EventArgs e)
        {
            (this.TopLevelControl as Form).WindowState = FormWindowState.Minimized;
        }

        private void sysBtnMaximize_Click(object sender, EventArgs e)
        {
            Form main = (this.TopLevelControl as Form);
            if (main.WindowState == FormWindowState.Maximized) main.WindowState = FormWindowState.Normal;
            else main.WindowState = FormWindowState.Maximized;
        }

        private void OnResize(object sender, EventArgs e)
        {
            Refresh();
        }

        private bool m_Dragging = false;

        private Point m_LastMousePos = new Point();

        private void OnMainBarMouseDown(object sender, MouseEventArgs e)
        {
            m_Dragging = true;
            m_LastMousePos = this.PointToScreen(e.Location);
        }

        private void OnMainBarMouseUp(object sender, MouseEventArgs e)
        {
            m_Dragging = false;
        }

        private void OnMainBarMouseMove(object sender, MouseEventArgs e)
        {
            if (m_Dragging)
            {
                Point eloc = this.PointToScreen(e.Location);
                int xdelta = eloc.X - m_LastMousePos.X;
                int ydelta = eloc.Y - m_LastMousePos.Y;
                m_LastMousePos = eloc;
                Point loc = (this.TopLevelControl as Form).Location;
                loc.Offset(xdelta, ydelta);
                (this.TopLevelControl as Form).Location = loc;
            }
        }

        private void OnMainBarPaint(object sender, PaintEventArgs e)
        {            
            //e.Graphics.FillRectangle(new SolidBrush(this.BackColor), this.ClientRectangle);
            if (m_CurrentMenu[0].UpMenu != null)
            {
                string text = "";
                foreach (MenuItem mi in m_CurrentMenu[0].UpMenu)
                    if (text.Length > 0) text += "|" + mi.Text;
                    else text = mi.Text;
                e.Graphics.DrawString(text, new Font("Segoe UI", 8), new System.Drawing.Drawing2D.LinearGradientBrush(new Point(0, 0), new Point(0, 16), /*Color.LightSteelBlue*/ Color.DodgerBlue, Color.White), 0.0f, 0.0f);
            }
        }

        private class MenuItem
        {
            public delegate void dOnMenuItem(EventArgs e);
            public delegate bool dMenuItemEnabler();
            public string Text;
            public object Action;
            public dMenuItemEnabler Enabler;
            public SySal.SySalNExTControls.SySalButton Button;
            public MenuItem [] UpMenu;
            public string ToolTipText;

            public MenuItem(string text, object action, dMenuItemEnabler enabler, string tooltiptext) 
            { 
                Text = text; 
                Action = action;
                Button = new SySal.SySalNExTControls.SySalButton();
                Button.Text = Text;
                Button.BackColor = Color.Transparent;
                Button.BackgroundImage = null;
                Button.AutoSize = true;
                Button.Visible = false;
                Button.ForeColor = Color.DodgerBlue;
                Button.FocusedColor = Color.Navy;
                Button.Font = new Font("Segoe UI", 14);
                Button.Tag = Action;
                Enabler = enabler;
                ToolTipText = tooltiptext;
                UpMenu = null;
            }

            private static bool sMenuItemEnablerFalse() { return false; }
            private static bool sMenuItemEnablerTrue() { return true; }
            public static dMenuItemEnabler MenuItemEnablerFalse { get { return new dMenuItemEnabler(sMenuItemEnablerFalse); } }
            public static dMenuItemEnabler MenuItemEnablerTrue { get { return new dMenuItemEnabler(sMenuItemEnablerTrue); } }
        }

        public void OnMenu_Up_Clicked(object sender, EventArgs e)
        {
            if (m_CurrentMenu[0].UpMenu != null)
            {
                m_CurrentMenu = m_CurrentMenu[0].UpMenu;
                SetCurrentMenu();
            }
        }

        public void OnMenu_Button_Clicked(object sender, EventArgs e)
        {
            SySal.SySalNExTControls.SySalButton btn = sender as SySal.SySalNExTControls.SySalButton;
            if (btn.Tag is MenuItem[])
            {
                m_CurrentMenu = btn.Tag as MenuItem[];
                SetCurrentMenu();
            }
            else if (btn.Tag != null && btn.Tag is MenuItem.dOnMenuItem)
                ((MenuItem.dOnMenuItem)btn.Tag).Invoke(e);
        }

        public void OnMenu_Setup_Scanner(EventArgs e)
        {
            TheScanner.EditMachineSettings(typeof(Scanner));
            EnableCurrentMenu();
        }

        public void OnMenu_Setup_Stage(EventArgs e)
        {
            if (TheScanner.iStage != null)            
                if (((object)TheScanner.iStage) is SySal.Management.IMachineSettingsEditor)
                {
                    SySal.Management.IMachineSettingsEditor iedt = ((SySal.Management.IMachineSettingsEditor)(object)TheScanner.iStage);
                    if (iedt.EditMachineSettings(iedt.GetType()))
                        /*ApplyStageSettings()*/;
                }
            EnableCurrentMenu();
        }                       

        public void OnMenu_Setup_Grabber(EventArgs e)
        {
            if (TheScanner.iGrab != null)            
                if (((object)TheScanner.iGrab) is SySal.Management.IMachineSettingsEditor)
                {
                    SySal.Management.IMachineSettingsEditor iedt = ((SySal.Management.IMachineSettingsEditor)(object)TheScanner.iGrab);
                    if (iedt.EditMachineSettings(iedt.GetType()))
                        DisplayMonitor("Grabber", TheScanner.iGrab);
                }
            EnableCurrentMenu();
        }

        public void OnMenu_Setup_GPU(EventArgs e)
        {
            TheScanner.EditGPUSettings();
            EnableCurrentMenu();
        }

        public void OnMenu_Setup_Imaging(EventArgs e)
        {
            TheScanner.RunImagingWizard();
            EnableCurrentMenu();
        }

        public void OnMenu_QuasiStaticAcquisition(EventArgs e)
        {
            TheScanner.RunQuasiStaticAcquisition();
            EnableCurrentMenu();
        }

        public void OnMenu_FullSpeedAcquisition(EventArgs e)
        {
            TheScanner.RunFullSpeedAcquisition();
            EnableCurrentMenu();
        }

        public void OnMenu_AcquireMarks(EventArgs e)
        {
            TheScanner.AcquireMarks();
            EnableCurrentMenu();
        }

        public void OnMenu_StartScanServer(EventArgs e)
        {
            TheScanner.StartScanServer();
            EnableCurrentMenu();
        }

        public void OnMenu_StopScanServer(EventArgs e)
        {
            TheScanner.StopScanServer();
            EnableCurrentMenu();
        }

        MenuItem[] MainMenu = null;

        private void OnMainFormLoad(object sender, EventArgs e)
        {
            TheScanner = new Scanner(this);
            SetCurrentMenu();
        }

        Scanner TheScanner = null;

        #region IScannerDataDisplay Members

        delegate void dDisplayStringAppend(string infotitle, string content);

        public void DisplayStringAppend(string infotitle, string content)
        {
            if (this.InvokeRequired) this.Invoke(new dDisplayStringAppend(this.Display), new object[] { infotitle, content });
            else
            {
                InfoPanel ipanel = null;
                foreach (Control ip in ClientPanel.Controls)
                    if (ip is InfoPanel && String.Compare((ip as InfoPanel).ContentTitle, infotitle, true) == 0)
                    {
                        ipanel = ip as InfoPanel;
                        break;
                    }
                if (ipanel == null) ipanel = new InfoPanel();
                ipanel.SetContent(infotitle, content, true);
                ipanel.Parent = ClientPanel;
                ipanel.AllowsClose = true;
                ipanel.AllowsExport = true;
                ipanel.AllowsRefreshContent = false;
                ClientPanel.Controls.Add(ipanel);
                MainToolTip.SetToolTip(ipanel.sysBtnExport, "Export content to a file.");
                MainToolTip.SetToolTip(ipanel.sysBtnRefresh, "Refresh content.");
                ipanel.Show();
                ipanel.BringToFront();
            }
        }

        delegate void dDisplay(string infotitle, object content);

        public void Display(string infotitle, object content)
        {
            if (this.InvokeRequired) this.Invoke(new dDisplay(this.Display), new object[] { infotitle, content });
            else
            {
                InfoPanel ipanel = null;
                foreach (Control ip in ClientPanel.Controls)
                    if (ip is InfoPanel && String.Compare((ip as InfoPanel).ContentTitle, infotitle, true) == 0)
                    {
                        ipanel = ip as InfoPanel;
                        break;
                    }
                if (ipanel == null) ipanel = new InfoPanel();
                ipanel.SetContent(infotitle, content);
                ipanel.Parent = ClientPanel;
                ipanel.AllowsClose = true;
                ipanel.AllowsExport = true;
                ipanel.AllowsRefreshContent = false;
                ClientPanel.Controls.Add(ipanel);
                MainToolTip.SetToolTip(ipanel.sysBtnExport, "Export content to a file.");
                MainToolTip.SetToolTip(ipanel.sysBtnRefresh, "Refresh content.");
                ipanel.Show();
                ipanel.BringToFront();
            }
        }

        public void DisplayMonitor(string infotitle, object monitoredobj)
        {
            if (this.InvokeRequired) this.Invoke(new dDisplay(this.DisplayMonitor), new object[] { infotitle, monitoredobj });
            else
            {
                InfoPanel ipanel = null;
                foreach (Control ip in ClientPanel.Controls)
                    if (ip is InfoPanel && String.Compare((ip as InfoPanel).ContentTitle, infotitle, true) == 0)
                    {
                        ipanel = ip as InfoPanel;
                        break;
                    }
                if (ipanel == null) ipanel = new InfoPanel();
                ipanel.SetContent(infotitle, (monitoredobj == null) ? "NULL" : monitoredobj);
                ipanel.Parent = ClientPanel;
                ipanel.AllowsClose = false;
                ipanel.AllowsExport = true;
                ipanel.AllowsRefreshContent = true;
                ClientPanel.Controls.Add(ipanel);
                MainToolTip.SetToolTip(ipanel.sysBtnExport, "Export content to a file.");
                MainToolTip.SetToolTip(ipanel.sysBtnRefresh, "Refresh content.");
                ipanel.Show();
                ipanel.BringToFront();
            }
        }

        delegate void dCloseMonitor(string infotitle);

        public void CloseMonitor(string infotitle)
        {
            if (this.InvokeRequired) this.Invoke(new dCloseMonitor(this.CloseMonitor), new object[] { infotitle });
            else
            {
                InfoPanel ipanel = null;
                foreach (Control ip in ClientPanel.Controls)
                    if (ip is InfoPanel && String.Compare((ip as InfoPanel).ContentTitle, infotitle, true) == 0)
                    {
                        ipanel = ip as InfoPanel;
                        break;
                    }
                if (ipanel != null)
                {
                    ClientPanel.Controls.Remove(ipanel);
                    ipanel.Close();
                    ipanel.Dispose();
                }
            }
        }

        #endregion

        private void OnTopPanelPaint(object sender, PaintEventArgs e)
        {
            e.Graphics.FillRectangle(new System.Drawing.Drawing2D.LinearGradientBrush(new Point(0, TopPanel.ClientRectangle.Top), new Point(0, TopPanel.ClientRectangle.Height), Color.White, Color.WhiteSmoke), TopPanel.ClientRectangle);
        }

        private delegate void dVoid();

        private void OnMainFormClosing(object sender, FormClosingEventArgs e)
        {
            new dVoid(TheScanner.Dispose).BeginInvoke(null, null);            
        }
    }
}