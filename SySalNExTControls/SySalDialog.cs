using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.SySalNExTControls
{
    public partial class SySalDialog : Form
    {
        [Description("Dialog caption."), Category("Appearance"), DefaultValue("Dialog"), Browsable(true), DesignerSerializationVisibility(DesignerSerializationVisibility.Visible)]
        public string DialogCaption
        {
            get { return TitleLabel.Text; }
            set { TitleLabel.Text = value; }
        }

        bool m_NoCloseButton = false;

        [Description("Allows hiding the \"close\" button."), Category("Appearance"), DefaultValue(false), Browsable(true), DesignerSerializationVisibility(DesignerSerializationVisibility.Visible)]
        public bool NoCloseButton
        {
            get { return m_NoCloseButton; }
            set 
            { 
                m_NoCloseButton = value;
                sysBtnClose.Visible = !m_NoCloseButton;
            }
        }

        public SySalDialog()
        {
            InitializeComponent();
            sysBtnClose.Top = 4;
            sysBtnClose.Left = ClientRectangle.Width - sysBtnClose.Width - 4;
            sysBtnClose.Visible = !m_NoCloseButton;
            TitleLabel.Text = "Dialog";
        }

        private void OnResize(object sender, EventArgs e)
        {
            sysBtnClose.Left = ClientRectangle.Width - sysBtnClose.Width - 4;
        }

        private bool m_Dragging = false;

        private Point m_LastMousePos = new Point();

        private void OnDialogMouseDown(object sender, MouseEventArgs e)
        {
            m_Dragging = true;
            m_LastMousePos = this.PointToScreen(e.Location);
            this.BringToFront();
        }

        private void OnDialogMouseMove(object sender, MouseEventArgs e)
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

        private void OnDialogMouseUp(object sender, MouseEventArgs e)
        {
            m_Dragging = false;
        }

        private void OnCloseClick(object sender, EventArgs e)
        {
            Close();
        }

        private void OnPaint(object sender, PaintEventArgs e)
        {
            e.Graphics.FillRectangle(new System.Drawing.Drawing2D.LinearGradientBrush(new Point(0, 0), new Point(0, this.Height), Color.White, Color.WhiteSmoke), ClientRectangle);
        }
    }
}