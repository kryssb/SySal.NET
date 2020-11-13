using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.NExTScanner
{
    public partial class InfoPanel : Form
    {
        [Description("Enables/disables the 'Close' button."), Category("Values"), DefaultValue(true), Browsable(true), DesignerSerializationVisibility(DesignerSerializationVisibility.Visible)]
        public bool AllowsClose
        {
            get { return sysBtnClose.Visible; }
            set
            {
                sysBtnClose.Visible = value;
            }
        }

        [Description("Enables/disables the 'Refresh' button."), Category("Values"), DefaultValue(false), Browsable(true), DesignerSerializationVisibility(DesignerSerializationVisibility.Visible)]
        public bool AllowsRefreshContent
        {
            get { return sysBtnRefresh.Visible; }
            set
            {
                sysBtnRefresh.Visible = value;
            }
        }

        [Description("Enables/disables the 'Export' button."), Category("Values"), DefaultValue(false), Browsable(true), DesignerSerializationVisibility(DesignerSerializationVisibility.Visible)]
        public bool AllowsExport
        {
            get { return sysBtnExport.Visible; }
            set
            {
                sysBtnExport.Visible = value;
            }
        }

        public InfoPanel()
        {
            InitializeComponent();
            TopLevel = false;
            sysBtnClose.Top = 4;
            sysBtnClose.Left = ClientRectangle.Width - sysBtnClose.Width - 4;
            sysBtnClose.Visible = true;
            sysBtnExport.Visible = true;
            sysBtnRefresh.Visible = false;            
        }

        private void OnResize(object sender, EventArgs e)
        {
            sysBtnClose.Left = ClientRectangle.Width - sysBtnClose.Width - 4;
        }

        private bool m_Dragging = false;

        private Point m_LastMousePos = new Point();

        private void OnTitleMouseDown(object sender, MouseEventArgs e)
        {
            m_Dragging = true;
            m_LastMousePos = this.PointToScreen(e.Location);
            this.BringToFront();
        }

        private void OnTitleMouseMove(object sender, MouseEventArgs e)
        {
            if (m_Dragging)
            {
                Point eloc = this.PointToScreen(e.Location);
                int xdelta = eloc.X - m_LastMousePos.X;
                int ydelta = eloc.Y - m_LastMousePos.Y;
                m_LastMousePos = eloc;
                Point loc = Location;
                loc.Offset(xdelta, ydelta);
                this.Location = loc;
            }
        }

        private void OnTitleMouseUp(object sender, MouseEventArgs e)
        {
            m_Dragging = false;
        }

        private void OnCloseClick(object sender, EventArgs e)
        {
            Close();
        }

        public string ContentTitle
        {
            get { return TitleLabel.Text; }
        }

        object m_Content = null;        

        public void SetContent(string title, object content, bool append = false)
        {
            TitleLabel.Text = title;
            ContentPanel.Controls.Clear();
            if (append && content is string && m_Content is string)
            {
                m_Content = ((string)m_Content) + "\r\n" + content;
            }
            else
            {
                m_Content = content;
            }
            if (content is Image)
            {
                PictureBox pb = new PictureBox();
                pb.BackColor = Color.Transparent;
                pb.Image = (Image)content;
                pb.SizeMode = PictureBoxSizeMode.Zoom;
                pb.Parent = ContentPanel;                
                pb.Dock = DockStyle.Fill;
            }
            else
            {
                TextBox tb = new TextBox();
                tb.BackColor = Color.LightGray;
                tb.Text = m_Content.ToString();                
                tb.Multiline = true;
                tb.ScrollBars = ScrollBars.Both;
                tb.Font = new Font("Lucida Console", 8);
                tb.Parent = ContentPanel;
                tb.Dock = DockStyle.Fill;
            }
        }

        private void OnPaint(object sender, PaintEventArgs e)
        {
            e.Graphics.FillRectangle(new System.Drawing.Drawing2D.LinearGradientBrush(new Point(0, 0), new Point(0, this.Height), Color.LightSteelBlue, Color.Snow), ClientRectangle);
        }

        private void OnRefreshClick(object sender, EventArgs e)
        {
            object r = RefreshContent();
            if (r != null) SetContent(TitleLabel.Text, r);
        }

        public virtual object RefreshContent()
        {
            return m_Content;
        }

        private void OnExportClick(object sender, EventArgs e)
        {
            Export();
        }

        public virtual void Export()
        {
            if (m_Content == null) return;
            try
            {
                if (m_Content is Image)
                {
                    SaveFileDialog sdlg = new SaveFileDialog();
                    sdlg.Title = "Select/create an image file.";
                    sdlg.Filter = "Base64 encoded files (*.b64)|*.b64|Windows Bitmap files (*.bmp)|*.bmp|Portable Network Graphics files (*.png)|*.png|Joint Photographic Experts Group files (*.jpg)|*.jpg|All files (*.*)|*.*";
                    if (sdlg.ShowDialog() == DialogResult.OK)
                    {
                        if (sdlg.FileName.ToLower().EndsWith(".b64"))
                        {                            
                            System.IO.File.WriteAllText(sdlg.FileName, SySal.Imaging.Base64ImageEncoding.ImageToBase64(new SySal.Executables.NExTScanner.SySalImageFromImage(m_Content as Image)));
                        }
                        else
                            ((Image)m_Content).Save(sdlg.FileName);
                    }
                }
                else                 
                {
                    SaveFileDialog sdlg = new SaveFileDialog();
                    sdlg.Title = "Select/create a text file.";
                    sdlg.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";
                    if (sdlg.ShowDialog() == DialogResult.OK)
                        System.IO.File.WriteAllText(sdlg.FileName, m_Content.ToString());
                }
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Error exporting data", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
    }
}