using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Text;
using System.Windows.Forms;

namespace SySal.SySalNExTControls
{
    [DefaultEvent("Click")]
    public partial class SySalButton : UserControl
    {
        private bool m_IsFocused = false;

        private Color m_FocusedColor = Color.Navy;

        [Description("Color of the text when the button has focus or the pointer enters."), Category("Values"), DefaultValue(""), Browsable(true), DesignerSerializationVisibility(DesignerSerializationVisibility.Visible)]
        public Color FocusedColor
        {
            get { return m_FocusedColor; }
            set
            {
                m_FocusedColor = value;
                Refresh();
            }
        }

        [Description("Button text."), Category("Values"), DefaultValue(""), Browsable(true), DesignerSerializationVisibility(DesignerSerializationVisibility.Visible)]
        public override string Text
        {
            get { return base.Text; }
            set
            {
                base.Text = value;
            }
        }

        SizeF TextSize;

        static System.Drawing.Image[] ClickImages = null;

        static SySalButton()
        {
            ClickImages = new Image[]
            {
                SySal.SySalNExTControls.SySalResources.circ_1,
                SySal.SySalNExTControls.SySalResources.circ_2,
                SySal.SySalNExTControls.SySalResources.circ_3,
                SySal.SySalNExTControls.SySalResources.circ_4,
                SySal.SySalNExTControls.SySalResources.circ_5,
                SySal.SySalNExTControls.SySalResources.circ_6,
                SySal.SySalNExTControls.SySalResources.circ_7,
                SySal.SySalNExTControls.SySalResources.circ_8,
                SySal.SySalNExTControls.SySalResources.circ_9
            };
        }

        protected System.Drawing.Image m_FromScreen;

        public SySalButton()
        {
            InitializeComponent();
            this.TextChanged += new System.EventHandler(OnTextChanged);
            m_FromScreen = new System.Drawing.Bitmap(ClickImages[0].Width, ClickImages[0].Height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
        }

        private void OnPaint(object sender, PaintEventArgs e)
        {
            if (this.BackColor != Color.Transparent) e.Graphics.Clear(this.BackColor);
            if (BackgroundImage != null) e.Graphics.DrawImage(BackgroundImage, this.ClientRectangle);
            if (this.Text != "") e.Graphics.DrawString(this.Text, this.Font, new SolidBrush(this.Enabled ? (m_IsFocused ? this.m_FocusedColor : this.ForeColor) : Color.LightGray), (this.ClientRectangle.Width - TextSize.Width) * 0.5f, (this.ClientRectangle.Height - TextSize.Height) * 0.5f);
        }

        private void OnMouseEnter(object sender, EventArgs e)
        {
            m_IsFocused = true;
            Refresh();
        }

        private void OnMouseLeave(object sender, EventArgs e)
        {
            m_IsFocused = false;
            Refresh();
        }

        private void OnFontChanged(object sender, EventArgs e)
        {
            TextSize = System.Drawing.Graphics.FromHwnd(this.Handle).MeasureString(Text, this.Font);
            if (AutoSize) AutoResize();
            Refresh();
        }

        private void OnTextChanged(object sender, EventArgs e)
        {
            TextSize = System.Drawing.Graphics.FromHwnd(this.Handle).MeasureString(Text, this.Font);
            if (AutoSize) AutoResize();
            Refresh();
        }

        private void AutoResize()
        {
            if (AutoSize && Text != "")
            {
                switch (Dock)
                {
                    case DockStyle.Left: Width = (int)TextSize.Width + 2; break;
                    case DockStyle.Right: Width = (int)TextSize.Width + 2; break;
                    case DockStyle.None:
                        Width = (int)TextSize.Width + 2;
                        Height = (int)TextSize.Height + 2;
                        break;
                    case DockStyle.Top: Height = (int)TextSize.Height + 2; break;
                    case DockStyle.Bottom: Height = (int)TextSize.Height + 2; break;
                }
            }
        }

        private void OnAutoSizeChanged(object sender, EventArgs e)
        {
            if (AutoSize) AutoResize();
        }

        public override Size GetPreferredSize(Size proposedSize)
        {
            Size availablearea = MinimumSize;
            if (Parent != null)
            {
                availablearea = Parent.ClientSize;
                int i;
                for (i = 0; i < Parent.Controls.Count && Parent.Controls[i] != this; i++)
                {
                    availablearea.Width -= Parent.Controls[i].Width;
                    availablearea.Height -= Parent.Controls[i].Height;
                }
                availablearea.Height -= Margin.Vertical;
                availablearea.Width -= Margin.Horizontal;
            }
            SizeF textsize = System.Drawing.Graphics.FromHwnd(this.Handle).MeasureString(Text, this.Font);
            Size sz = new Size((int)textsize.Width + 2, (int)textsize.Height + 2);
            switch (Dock)
            {
                case DockStyle.Left: if (Text != "") sz.Width = (int)TextSize.Width + 2; sz.Height = availablearea.Height; break;
                case DockStyle.Right: if (Text != "") sz.Width = (int)TextSize.Width + 2; sz.Height = availablearea.Height; break;
                case DockStyle.None:
                    if (Text != "")
                    {
                        sz.Width = (int)TextSize.Width + 2;
                        sz.Height = (int)TextSize.Height + 2;
                    }
                    break;
                case DockStyle.Top: if (Text != "") sz.Height = (int)TextSize.Height + 2; sz.Width = availablearea.Width; break;
                case DockStyle.Bottom: if (Text != "") sz.Height = (int)TextSize.Height + 2; sz.Width = availablearea.Width; break;
                case DockStyle.Fill: sz = availablearea; break;
            }
            return sz;
        }

        private void OnClick(object sender, EventArgs e)
        {
            MouseEventArgs me = (MouseEventArgs)e;
            System.Windows.Forms.PictureBox pb = new PictureBox();
            pb.Visible = false;
            System.Drawing.Point p1 = PointToScreen(me.Location);
            pb.BackColor = Color.Transparent;
            pb.Parent = this.TopLevelControl;
            pb.BringToFront();
            System.Drawing.Point p = pb.Parent.PointToClient(p1);
            pb.Left = p.X - ClickImages[0].Width / 2;
            pb.Width = ClickImages[0].Width;
            pb.Top = p.Y - ClickImages[0].Height / 2;
            pb.Height = ClickImages[0].Height;
            int i;
            System.Drawing.Graphics g1 = System.Drawing.Graphics.FromImage(m_FromScreen);
            g1.CopyFromScreen(p1.X - ClickImages[0].Width / 2, p1.Y - ClickImages[0].Height / 2, 0, 0, new Size(ClickImages[0].Width, ClickImages[0].Height));
            g1.Dispose();
            //pb.Show();
            for (i = 0; i < ClickImages.Length; i++)
            {
                System.Drawing.Graphics g = System.Drawing.Graphics.FromHwnd(pb.Handle);
                g.DrawImageUnscaled(m_FromScreen, 0, 0);
                g.DrawImageUnscaled(ClickImages[i], 0, 0);
                System.Threading.Thread.Sleep(50);
                g.Dispose();
                if (i == 0) pb.Show();
            }
            pb.Hide();
            pb.Dispose();
        }
    }
}
