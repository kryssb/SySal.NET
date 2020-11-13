using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.EasyReconstruct
{
    internal interface IDecaySearchAutomation
    {
        int CurrentStep { get; set; }
        int[] PrimaryVertex { get; }
        string ErrorsInFeedbackTracks { get; }

        void FindPrimaryTracks(string postols, string slopetols);
        void FindPrimaryVertex(string postols, string slopetols);
        void ComputeMomenta(string postols, string slopetols);
        void MergeManualTracks(string postols, string slopetols);
        void BrowseTrack(int id);
        int [] CheckPrimaryTracksHoles();
        void IsolatedTrackEventExtraTrackSearch();
        void OneMuOrMultiProngZeroMuEventExtraTrackSearch();
        void ZeroMu123ProngEventExtraTrackSearch();
    }

    internal partial class DecaySearchAssistantForm : Form
    {        
        IDecaySearchAutomation iAuto;

        #region checks

        public bool CheckFeedbackTracks()
        {
            if (CheckPrimaryVertexSet() == false) return false;
            string vstr = iAuto.ErrorsInFeedbackTracks;
            if (vstr.Trim().Length > 0)
            {
                new QBrowser("Invalid feedback information", vstr).ShowDialog();
                return false;
            }
            return true;
        }

        public bool CheckPrimaryVertexSet()
        {
            int [] vi = iAuto.PrimaryVertex;
            if (vi.Length != 1)
            {
                string vstr = "";
                if (vi.Length == 0) vstr = "No vertex flagged as primary";
                else
                {
                    vstr = "Ids of vertices flagged as primary:";
                    foreach (int i in vi)
                        vstr += ("\r\n" + i);
                }
                MessageBox.Show(vstr, "Primary Vertex needed", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return false;
            }
            return true;
        }

        public bool CheckPrimaryTracksQuality()
        {
            if (CheckPrimaryVertexSet() == false) return false;
            int[] htk = iAuto.CheckPrimaryTracksHoles();
            int nmand = 0;
            foreach (int h in htk)
                if (h < 0)
                {
                    nmand++;
                    iAuto.BrowseTrack(~h);
                }            
            if (htk.Length > nmand && MessageBox.Show("One or more tracks miss base tracks in the most upstream plates.\r\nBrowse them to add manual checks?", "Warning/Confirmation", MessageBoxButtons.YesNo, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2) == DialogResult.Yes)
            {
                foreach (int h in htk)
                    if (h >= 0)
                        iAuto.BrowseTrack(h);
            }
            if (htk.Length > 0)
            {
                string ot = "";
                if (nmand > 0)
                {
                    ot += "Tracks that need to be filled to perform in-track decay search:";
                    foreach (int h in htk)
                        if (h < 0)
                            ot += "\r\n" + (~h);
                }
                if (htk.Length > nmand)
                {
                    if (ot.Length > 0) ot += "\r\n";
                    ot += "Tracks that are missing base tracks in the first four plates:";
                    foreach (int h in htk)
                        if (h >= 0)
                            ot += "\r\n" + h;
                }
                MessageBox.Show(ot, "Faults found in quality check", MessageBoxButtons.OK, nmand > 0 ? MessageBoxIcon.Error : MessageBoxIcon.Warning);
            }
            return true;
        }

        #endregion


        public DecaySearchAssistantForm(IDecaySearchAutomation iauto)
        {
            InitializeComponent();
            iAuto = iauto;
        }

        string m_CurrentPage = "";

        string Style
        {
            get
            {
                return
                    "<style type=\"text/css\">" +
                    " body { font-family:Trebuchet MS,Arial,Helvetica; font-size:" + m_FontSize + "px }" +
                    " h1 { font-size:" + (m_FontSize + 4) + "px; color:teal }" +
                    " h2 { font-size:" + (m_FontSize + 2) + "px; color:navy }" +
                    " .doforme { font-size:" + m_FontSize + "px; border-width:thin; border-color:navy; text-align:right; }" +
                    "</style>";
            }
        }       

        void LoadPage(string page)
        {
            System.IO.StreamReader myStream;
            System.Reflection.Assembly myAssembly = System.Reflection.Assembly.GetExecutingAssembly();
            myStream = new System.IO.StreamReader(myAssembly.GetManifestResourceStream("EasyReconstruct." + page));
            m_CurrentPage = page;
            string pagetext = myStream.ReadToEnd();
            webBrowser1.DocumentText = pagetext.Replace("<MYSTYLE />", Style);
        }

        private void OnLoad(object sender, EventArgs e)
        {
            Left = System.Windows.Forms.Screen.PrimaryScreen.Bounds.Right - Size.Width;
            LoadPage("DSApage1_1.htm");
        }

        const string CheckStr = "chk=";

        const string DoStr = "do=";        

        private void OnNewPage(object sender, WebBrowserNavigatingEventArgs e)
        {            
            string [] q = e.Url.Query.Split('&','?');            
            foreach (string qs in q)
            {                
                if (qs.StartsWith(CheckStr))
                {
                    string method = qs.Substring(CheckStr.Length);
                    try
                    {
                        if (Convert.ToBoolean(this.GetType().GetMethod(method).Invoke(this, new object[0])) == false)
                        {
                            e.Cancel = true;
                            return;
                        }
                    }
                    catch (Exception x) 
                    {
                        MessageBox.Show(x.ToString(), "Programming error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        e.Cancel = true; 
                        return;
                    }
                }
                else if (qs.StartsWith(DoStr))
                {
                    string method = qs.Substring(DoStr.Length);
                    int i = 0;
                    string wp = webBrowser1.DocumentText;
                    System.Collections.ArrayList ar = new System.Collections.ArrayList();
                    while (true)
                    {
                        try
                        {
                            ar.Add("" + webBrowser1.Document.GetElementById("p" + i++).GetAttribute("value"));
                        }
                        catch (Exception) { break; }
                    }
                    try
                    {
                        iAuto.GetType().GetMethod(method).Invoke(iAuto, (string [])ar.ToArray(typeof(string)));
                    }
                    catch (Exception x)
                    {
                        MessageBox.Show(x.ToString(), "Programming error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                    e.Cancel = true;
                    return;
                }
            }
            LoadPage(e.Url.AbsolutePath);            
        }

        int m_FontSize = 12;

        private void OnFontSizeChanged(object sender, EventArgs e)
        {
            try
            {
                int fs = Convert.ToInt32(txtFontSize.Text);
                if (fs < 8 || fs > 40) return;
                m_FontSize = fs;
                LoadPage(m_CurrentPage);
            }
            catch (Exception)
            {
                
            }
        }
    }
}