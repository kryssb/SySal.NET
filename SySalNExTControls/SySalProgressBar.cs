using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Text;
using System.Windows.Forms;

namespace SySal.SySalNExTControls
{
    public enum SySalProgressBarDirection { BottomUp, TopDown, LeftToRight, RightToLeft }

    public partial class SySalProgressBar : UserControl
    {
        private SySalProgressBarDirection m_Direction = SySalProgressBarDirection.BottomUp;

        [Description("Filling direction of the progress bar."), Category("Appearance"), DefaultValue(SySalProgressBarDirection.BottomUp), Browsable(true), DesignerSerializationVisibility(DesignerSerializationVisibility.Visible)]
        public SySalProgressBarDirection Direction
        {
            get { return m_Direction; }
            set
            {
                m_Direction = value;
                Refresh();
            }
        }

        private SySalProgressBarDirection m_GradientDirection = SySalProgressBarDirection.BottomUp;

        [Description("Gradient direction of the progress bar."), Category("Appearance"), DefaultValue(SySalProgressBarDirection.BottomUp), Browsable(true), DesignerSerializationVisibility(DesignerSerializationVisibility.Visible)]
        public SySalProgressBarDirection GradientDirection
        {
            get { return m_GradientDirection; }
            set
            {
                m_GradientDirection = value;
                Refresh();
            }
        }

        Color[] m_EmptyGradientColors = new Color[] { Color.Lavender, Color.Azure, Color.Lavender };

        [Description("Colors for the gradient of the empty part of the progress bar."), Category("Appearance"), DefaultValue(""), Browsable(true), DesignerSerializationVisibility(DesignerSerializationVisibility.Visible)]
        public Color[] EmptyGradientColors
        {
            get { return m_EmptyGradientColors; }
            set
            {
                Color[] ng = new Color[Math.Max(2, value.Length)];
                int i;
                Color LastColor = Color.LightGray;
                for (i = 0; i < ng.Length; i++)
                    ng[i] = LastColor;
                for (i = 0; i < Math.Min(value.Length, ng.Length); i++)                
                    ng[i] = LastColor = value[i];
                while (i < ng.Length) ng[i++] = LastColor;
                m_EmptyGradientColors = value;
                if (m_EmptyGradientStops.Length != m_EmptyGradientColors.Length - 2)
                {
                    double[] stops = new double[m_EmptyGradientColors.Length - 2];
                    for (i = 0; i < stops.Length; i++)
                        stops[i] = (1.0 / (m_EmptyGradientColors.Length - 1)) * (i + 1);
                    m_EmptyGradientStops = stops;
                }
                Refresh();
            }
        }

        double[] m_EmptyGradientStops = new double[] { 0.5 };

        [Description("Stops for the gradient of the empty part of the progress bar."), Category("Appearance"), DefaultValue(new double[] { 0.5 }), Browsable(true), DesignerSerializationVisibility(DesignerSerializationVisibility.Visible)]
        public double [] EmptyGradientStops
        {
            get { return m_EmptyGradientStops; }
            set
            {
                foreach (double d in value)
                    if (d < 0.0 || d > 1.0)
                        throw new Exception("All stops must be in the range 0.0-1.0.");
                int i;
                for (i = 1; i < value.Length; i++)
                    if (value[i] <= value[i - 1])
                        throw new Exception("Stops must be sorted in ascending order and duplicate values are not allowed.");
                m_EmptyGradientStops = value;
                if (m_EmptyGradientColors.Length < m_EmptyGradientStops.Length + 2)
                {
                    Color[] ng = new Color[m_EmptyGradientStops.Length + 2];
                    m_EmptyGradientColors.CopyTo(ng, 0);
                    for (i = m_EmptyGradientColors.Length; i < m_EmptyGradientStops.Length + 2; i++)
                        ng[i] = m_EmptyGradientColors[m_EmptyGradientColors.Length - 1];
                    m_EmptyGradientColors = ng;
                }
                else if (m_EmptyGradientColors.Length > m_EmptyGradientStops.Length + 2)
                {
                    Color[] ng = new Color[m_EmptyGradientStops.Length + 2];
                    for (i = 0; i < m_EmptyGradientStops.Length + 1; i++)
                        ng[i] = m_EmptyGradientColors[i];
                    ng[i] = m_EmptyGradientColors[m_EmptyGradientColors.Length - 1];
                    m_EmptyGradientColors = ng;
                }
                Refresh();
            }
        }

        Color[] m_FillGradientColors = new Color[] { Color.PowderBlue, Color.DodgerBlue, Color.PowderBlue };

        [Description("Colors for the gradient of the filled part of the progress bar."), Category("Appearance"), DefaultValue(""), Browsable(true), DesignerSerializationVisibility(DesignerSerializationVisibility.Visible)]
        public Color[] FillGradientColors
        {
            get { return m_FillGradientColors; }
            set
            {
                Color[] ng = new Color[Math.Max(2, value.Length)];
                int i;
                Color LastColor = Color.LightGray;
                for (i = 0; i < ng.Length; i++)
                    ng[i] = LastColor;
                for (i = 0; i < Math.Min(value.Length, ng.Length); i++)
                    ng[i] = LastColor = value[i];
                while (i < ng.Length) ng[i++] = LastColor;
                m_FillGradientColors = value;
                if (m_FillGradientStops.Length != m_FillGradientColors.Length - 2)
                {
                    double[] stops = new double[m_FillGradientColors.Length - 2];
                    for (i = 0; i < stops.Length; i++)
                        stops[i] = (1.0 / (m_FillGradientColors.Length - 1)) * (i + 1);
                    m_FillGradientStops = stops;
                }
                Refresh();
            }
        }

        double[] m_FillGradientStops = new double[] { 0.5 };

        [Description("Stops for the gradient of the filled part of the progress bar."), Category("Appearance"), DefaultValue(new double[] { 0.5 }), Browsable(true), DesignerSerializationVisibility(DesignerSerializationVisibility.Visible)]
        public double[] FillGradientStops
        {
            get { return m_FillGradientStops; }
            set
            {
                foreach (double d in value)
                    if (d < 0.0 || d > 1.0)
                        throw new Exception("All stops must be in the range 0.0-1.0.");
                int i;
                for (i = 1; i < value.Length; i++)
                    if (value[i] <= value[i - 1])
                        throw new Exception("Stops must be sorted in ascending order and duplicate values are not allowed.");
                m_FillGradientStops = value;
                if (m_FillGradientColors.Length < m_FillGradientStops.Length + 2)
                {
                    Color[] ng = new Color[m_FillGradientStops.Length + 2];
                    m_FillGradientColors.CopyTo(ng, 0);
                    for (i = m_FillGradientColors.Length; i < m_FillGradientStops.Length + 2; i++)
                        ng[i] = m_FillGradientColors[m_FillGradientColors.Length - 1];
                    m_FillGradientColors = ng;
                }
                else if (m_FillGradientColors.Length > m_FillGradientStops.Length + 2)
                {
                    Color[] ng = new Color[m_FillGradientStops.Length + 2];
                    for (i = 0; i < m_FillGradientStops.Length + 1; i++)
                        ng[i] = m_FillGradientColors[i];
                    ng[i] = m_FillGradientColors[m_FillGradientColors.Length - 1];
                    m_FillGradientColors = ng;
                }
                Refresh();
            }
        }

        private double m_Value = 0.0;

        [Description("Value of the progress bar."), Category("Values"), DefaultValue(0.0), Browsable(true), DesignerSerializationVisibility(DesignerSerializationVisibility.Visible)]
        public double Value
        {
            get { return m_Value; }
            set
            {
                m_Value = Math.Min(Math.Max(m_Minimum, value), m_Maximum);
                Refresh();
            }
        }

        private double m_Minimum = 0.0;

        [Description("Minimum value of the progress bar."), Category("Values"), DefaultValue(0.0), Browsable(true), DesignerSerializationVisibility(DesignerSerializationVisibility.Visible)]
        public double Minimum
        {
            get { return m_Minimum; }
            set
            {
                m_Minimum = value;                
            }
        }

        private double m_Maximum = 100.0;

        [Description("Maximum value of the progress bar."), Category("Values"), DefaultValue(100.0), Browsable(true), DesignerSerializationVisibility(DesignerSerializationVisibility.Visible)]
        public double Maximum
        {
            get { return m_Maximum; }
            set
            {
                m_Maximum = value;                
            }
        }

        public SySalProgressBar()
        {
            InitializeComponent();
        }

        private void OnPaint(object sender, PaintEventArgs e)
        {
            double[] eminstops = new double[m_EmptyGradientStops.Length + 1];
            double[] emaxstops = new double[m_EmptyGradientStops.Length + 1];
            m_EmptyGradientStops.CopyTo(eminstops, 1);
            eminstops[0] = 0.0;
            m_EmptyGradientStops.CopyTo(emaxstops, 0);
            emaxstops[emaxstops.Length - 1] = 1.0;
            double[] fminstops = new double[m_FillGradientStops.Length + 1];
            double[] fmaxstops = new double[m_FillGradientStops.Length + 1];
            m_FillGradientStops.CopyTo(fminstops, 1);
            fminstops[0] = 0.0;
            m_FillGradientStops.CopyTo(fmaxstops, 0);
            fmaxstops[fmaxstops.Length - 1] = 1.0;
            int i;
            PointF a1 = new PointF();
            PointF a = new PointF();
            PointF b = new PointF();
            PointF b1 = new PointF();
            Rectangle barrect = e.ClipRectangle;
            switch (m_Direction)
            {
                case SySalProgressBarDirection.BottomUp:
                    barrect.Intersect(new Rectangle(0, Height - (int)Math.Round((Value / (Maximum - Minimum)) * Height), Width, Height));
                    break;

                case SySalProgressBarDirection.TopDown:
                    barrect.Intersect(new Rectangle(0, 0, Width, (int)Math.Round((Value / (Maximum - Minimum)) * Height)));
                    break;

                case SySalProgressBarDirection.LeftToRight:
                    barrect.Intersect(new Rectangle(0, 0, (int)Math.Round((Value / (Maximum - Minimum)) * Width), Height));
                    break;

                case SySalProgressBarDirection.RightToLeft:
                    barrect.Intersect(new Rectangle(Width - (int)Math.Round((Value / (Maximum - Minimum)) * Width), 0, Width, Height));
                    break;
            }
            switch (m_GradientDirection)
            {
                case SySalProgressBarDirection.TopDown:
                    for (i = 0; i < eminstops.Length; i++)
                    {
                        a.Y = (float)Math.Round(ClientRectangle.Height * eminstops[i]);
                        b.Y = (float)Math.Round(ClientRectangle.Height * emaxstops[i]);
                        a1.Y = a.Y - 1.0f;
                        b1.Y = b.Y + 1.0f;
                        e.Graphics.FillRectangle(new System.Drawing.Drawing2D.LinearGradientBrush(a1, b1, m_EmptyGradientColors[i], m_EmptyGradientColors[i + 1]), 0, a.Y, Width, b.Y - a.Y);
                    }
                    e.Graphics.SetClip(barrect);
                    for (i = 0; i < fminstops.Length; i++)
                    {
                        a.Y = (float)Math.Round(ClientRectangle.Height * fminstops[i]);
                        b.Y = (float)Math.Round(ClientRectangle.Height * fmaxstops[i]);
                        a1.Y = a.Y - 1.0f;
                        b1.Y = b.Y + 1.0f;
                        e.Graphics.FillRectangle(new System.Drawing.Drawing2D.LinearGradientBrush(a1, b1, m_FillGradientColors[i], m_FillGradientColors[i + 1]), 0, a.Y, Width, b.Y - a.Y);
                    }
                    break;
                
                case SySalProgressBarDirection.BottomUp:
                    for (i = 0; i < eminstops.Length; i++)
                    {
                        a.Y = (float)Math.Round(ClientRectangle.Height * (1.0 - eminstops[i]));
                        b.Y = (float)Math.Round(ClientRectangle.Height * (1.0 - emaxstops[i]));
                        a1.Y = a.Y + 1.0f;
                        b1.Y = b.Y - 1.0f;
                        e.Graphics.FillRectangle(new System.Drawing.Drawing2D.LinearGradientBrush(a1, b1, m_EmptyGradientColors[i], m_EmptyGradientColors[i + 1]), 0, b.Y, Width, a.Y - b.Y);
                    }
                    e.Graphics.SetClip(barrect); for (i = 0; i < fminstops.Length; i++)
                    {
                        a.Y = (float)Math.Round(ClientRectangle.Height * (1.0 - fminstops[i]));
                        b.Y = (float)Math.Round(ClientRectangle.Height * (1.0 - fmaxstops[i]));
                        a1.Y = a.Y + 1.0f;
                        b1.Y = b.Y - 1.0f;
                        e.Graphics.FillRectangle(new System.Drawing.Drawing2D.LinearGradientBrush(a1, b1, m_FillGradientColors[i], m_FillGradientColors[i + 1]), 0, b.Y, Width, a.Y - b.Y);
                    }
                    break;

                case SySalProgressBarDirection.LeftToRight:
                    for (i = 0; i < eminstops.Length; i++)
                    {
                        a.X = (float)Math.Round(ClientRectangle.Width * eminstops[i]);
                        b.X = (float)Math.Round(ClientRectangle.Width * emaxstops[i]);
                        a1.X = a.X - 1.0f;
                        b1.X = b.X + 1.0f;
                        e.Graphics.FillRectangle(new System.Drawing.Drawing2D.LinearGradientBrush(a1, b1, m_EmptyGradientColors[i], m_EmptyGradientColors[i + 1]), a.X, 0, b.X - a.X, Height);
                    }
                    e.Graphics.SetClip(barrect); 
                    for (i = 0; i < fminstops.Length; i++)
                    {
                        a.X = (float)Math.Round(ClientRectangle.Width * fminstops[i]);
                        b.X = (float)Math.Round(ClientRectangle.Width * fmaxstops[i]);
                        a1.X = a.X - 1.0f;
                        b1.X = b.X + 1.0f;
                        e.Graphics.FillRectangle(new System.Drawing.Drawing2D.LinearGradientBrush(a1, b1, m_FillGradientColors[i], m_FillGradientColors[i + 1]), a.X, 0, b.X - a.X, Height);
                    }
                    break;

                case SySalProgressBarDirection.RightToLeft:
                    for (i = 0; i < eminstops.Length; i++)
                    {
                        a.X = (float)Math.Round(ClientRectangle.Width * (1.0 - eminstops[i]));
                        b.X = (float)Math.Round(ClientRectangle.Width * (1.0 - emaxstops[i])) + 1;
                        a1.X = a.X + 1.0f;
                        b1.X = b.X - 1.0f;
                        e.Graphics.FillRectangle(new System.Drawing.Drawing2D.LinearGradientBrush(a1, b1, m_EmptyGradientColors[i], m_EmptyGradientColors[i + 1]), b.X, 0, a.X - b.X, Height);
                    }
                    e.Graphics.SetClip(barrect); 
                    for (i = 0; i < fminstops.Length; i++)
                    {
                        a.X = (float)Math.Round(ClientRectangle.Width * (1.0 - fminstops[i]));
                        b.X = (float)Math.Round(ClientRectangle.Width * (1.0 - fmaxstops[i])) + 1;
                        a1.X = a.X + 1.0f;
                        b1.X = b.X - 1.0f;
                        e.Graphics.FillRectangle(new System.Drawing.Drawing2D.LinearGradientBrush(a1, b1, m_FillGradientColors[i], m_FillGradientColors[i + 1]), b.X, 0, a.X - b.X, Height);
                    }
                    break;
            }
        }
    }
}
