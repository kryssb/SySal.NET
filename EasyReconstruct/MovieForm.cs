using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.EasyReconstruct
{
    public partial class MovieForm : Form
    {
        internal GDI3D.Control.GDIDisplay gdiDisplay;

        internal long m_Event;

        internal int m_Brick;

        public MovieForm(GDI3D.Control.GDIDisplay disp, long ev, int bk)
        {
            InitializeComponent();
            gdiDisplay = disp;
            m_Event = ev;
            m_Brick = bk;
            InitDataMap();
        }

        private void btnMake_Click(object sender, EventArgs e)
        {
            SaveFileDialog sdlg = new SaveFileDialog();
            sdlg.Title = "Select movie file";
            sdlg.Filter = "Animated GIF (*.gif)|*.gif";
            if (sdlg.ShowDialog() != DialogResult.OK) return;
            double DX = 0.0, DY = 0.0, DZ = 0.0;
            double NX = 0.0, NY = 0.0, NZ = 0.0;
            double BX = 0.0, BY = 0.0, BZ = 0.0;
            double PX = 0.0, PY = 0.0, PZ = 0.0;
            gdiDisplay.GetCameraOrientation(ref DX, ref DY, ref DZ, ref NX, ref NY, ref NZ, ref BX, ref BY, ref BZ);
            gdiDisplay.GetCameraSpotting(ref PX, ref PY, ref PZ);
            Cursor oldc = Cursor;
            try
            {
                Cursor = Cursors.WaitCursor;                
                gdiDisplay.AutoRender = false;
                Bitmap b = new Bitmap(gdiDisplay.Width, gdiDisplay.Height);
                GDI3D.Movie mv = new GDI3D.Movie(1);
                mv.Comment = txtComments.Text;
                int i;
                Color bkcol = gdiDisplay.BackColor;
                Color textcol = Color.FromArgb((bkcol.R + 128) % 256, (bkcol.G + 128) % 256, (bkcol.B + 128) % 256);
                System.Collections.ArrayList ar = new System.Collections.ArrayList();
                SySal.BasicTypes.Vector startdir = new SySal.BasicTypes.Vector();
                SySal.BasicTypes.Vector startnorm = new SySal.BasicTypes.Vector();
                SySal.BasicTypes.Vector finishdir = new SySal.BasicTypes.Vector();
                SySal.BasicTypes.Vector finishnorm = new SySal.BasicTypes.Vector();
                SySal.BasicTypes.Vector loopdir = new SySal.BasicTypes.Vector();
                SySal.BasicTypes.Vector loopnorm = new SySal.BasicTypes.Vector();                
                if (chkViewXY.Checked)
                {
                    loopdir.X = 0.0; loopdir.Y = 0.0; loopdir.Z = -1.0;
                    loopnorm.X = 0.0; loopnorm.Y = -1.0; loopnorm.Z = 0.0;
                }
                else if (chkViewXZ.Checked)
                {
                    loopdir.X = 0.0; loopdir.Y = 1.0; loopdir.Z = 0.0;
                    loopnorm.X = 0.0; loopnorm.Y = 0.0; loopnorm.Z = -1.0;
                }
                else if (chkViewYZ.Checked)
                {
                    loopdir.X = -1.0; loopdir.Y = 0.0; loopdir.Z = 0.0;
                    loopnorm.X = 0.0; loopnorm.Y = 0.0; loopnorm.Z = -1.0;
                }
                else if (chkViewYRot.Checked)
                {
                    loopdir.X = 0.0; loopdir.Y = 0.0; loopdir.Z = -1.0;
                    loopnorm.X = 0.0; loopnorm.Y = -1.0; loopnorm.Z = 0.0;
                }
                else if (chkViewXYPan.Checked)
                {
                    loopdir.X = 0.0; loopdir.Y = 0.0; loopdir.Z = -1.0;
                    loopnorm.X = 0.0; loopnorm.Y = -1.0; loopnorm.Z = 0.0;
                }
                else
                {
                    MessageBox.Show("At least one set of views must be selected.", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
                
                startdir = loopdir;
                startnorm = loopnorm;
                if (chkViewXY.Checked)
                {
                    object[] o = new object[4];                    
                    o[0] = startdir;
                    o[1] = startnorm;
                    o[2] = startdir ^ startnorm;
                    o[3] = "XY";
                    for (i = 0; i < (int)DataMap[0, 1]; i++)
                        ar.Add(o);
                }
                if (chkViewXZ.Checked)
                {
                    finishdir.X = 0.0; finishdir.Y = 1.0; finishdir.Z = 0.0;
                    finishnorm.X = 0.0; finishnorm.Y = 0.0; finishnorm.Z = -1.0;
                    InterpolateCamera(startdir, startnorm, finishdir, finishnorm, 90 / trkSpeed.Value, 1.0, ar, "");
                    startdir = finishdir;
                    startnorm = finishnorm;
                    object[] o = new object[4];
                    o[0] = startdir;
                    o[1] = startnorm;
                    o[2] = startdir ^ startnorm;
                    o[3] = "XZ";
                    for (i = 0; i < (int)DataMap[1, 1]; i++)
                        ar.Add(o);
                }
                if (chkViewYZ.Checked)
                {
                    finishdir.X = -1.0; finishdir.Y = 0.0; finishdir.Z = 0.0;
                    finishnorm.X = 0.0; finishnorm.Y = 0.0; finishnorm.Z = -1.0;
                    InterpolateCamera(startdir, startnorm, finishdir, finishnorm, 90 / trkSpeed.Value, 1.0, ar, "");
                    startdir = finishdir;
                    startnorm = finishnorm;
                    object[] o = new object[4];
                    o[0] = startdir;
                    o[1] = startnorm;
                    o[2] = startdir ^ startnorm;
                    o[3] = "YZ";
                    for (i = 0; i < (int)DataMap[2, 1]; i++)
                        ar.Add(o);
                }
                if (chkViewYRot.Checked)
                {
                    finishdir.X = 0.0; finishdir.Y = 0.0; finishdir.Z = -1.0;
                    finishnorm.X = 0.0; finishnorm.Y = -1.0; finishnorm.Z = 0.0;
                    InterpolateCamera(startdir, startnorm, finishdir, finishnorm, 90 / trkSpeed.Value, 1.0, ar, "");
                    startdir.X = 1.0; startdir.Y = 0.0; startdir.Z = 0.0;
                    InterpolateCamera(finishdir, finishnorm, startdir, finishnorm, (int)DataMap[3, 1], 0.0, ar, "Y rot");
                    ar.RemoveAt(ar.Count - 1);
                    finishdir.X = 0.0; finishdir.Y = 0.0; finishdir.Z = 1.0;
                    InterpolateCamera(startdir, finishnorm, finishdir, finishnorm, (int)DataMap[3, 1], 0.0, ar, "Y rot");
                    ar.RemoveAt(ar.Count - 1);
                    startdir.X = -1.0; startdir.Y = 0.0; startdir.Z = 0.0;
                    InterpolateCamera(finishdir, finishnorm, startdir, finishnorm, (int)DataMap[3, 1], 0.0, ar, "Y rot");
                    ar.RemoveAt(ar.Count - 1);
                    finishdir.X = 0.0; finishdir.Y = 0.0; finishdir.Z = -1.0;
                    InterpolateCamera(startdir, finishnorm, finishdir, finishnorm, (int)DataMap[3, 1], 0.0, ar, "Y rot");
                    startdir = finishdir; startnorm = finishnorm;
                }
                if (chkViewXYPan.Checked)
                {
/*
                    finishdir.X = -0.3; finishdir.Y = -0.3; finishdir.Z = -1.0; finishdir = finishdir.UnitVector;
                    finishnorm.X = 0.0; finishnorm.Y = -1.0; finishnorm.Z = 0.0; finishnorm = (finishnorm - (finishnorm * finishdir) * finishdir).UnitVector;
                    InterpolateCamera(startdir, startnorm, finishdir, finishnorm, 90 / trkSpeed.Value, 1.0, ar, "XY pan");

                    startdir = finishdir; startnorm = finishnorm;
                    finishdir.X = 0.3; finishdir.Y = -0.3; finishdir.Z = -1.0; finishdir = finishdir.UnitVector;
                    finishnorm.X = 0.0; finishnorm.Y = -1.0; finishnorm.Z = 0.0; finishnorm = (finishnorm - (finishnorm * finishdir) * finishdir).UnitVector;
                    InterpolateCamera(startdir, startnorm, finishdir, finishnorm, (int)DataMap[4, 1], 1.0, ar, "XY pan");
                    
                    startdir = finishdir; startnorm = finishnorm;
                    finishdir.X = 0.3; finishdir.Y = 0.3; finishdir.Z = -1.0; finishdir = finishdir.UnitVector;
                    finishnorm.X = 0.0; finishnorm.Y = -1.0; finishnorm.Z = 0.0; finishnorm = (finishnorm - (finishnorm * finishdir) * finishdir).UnitVector;
                    InterpolateCamera(startdir, startnorm, finishdir, finishnorm, (int)DataMap[4, 1], 1.0, ar, "XY pan");

                    startdir = finishdir; startnorm = finishnorm;
                    finishdir.X = -0.3; finishdir.Y = 0.3; finishdir.Z = -1.0; finishdir = finishdir.UnitVector;
                    finishnorm.X = 0.0; finishnorm.Y = -1.0; finishnorm.Z = 0.0; finishnorm = (finishnorm - (finishnorm * finishdir) * finishdir).UnitVector;
                    InterpolateCamera(startdir, startnorm, finishdir, finishnorm, (int)DataMap[4, 1], 1.0, ar, "XY pan");

                    startdir = finishdir; startnorm = finishnorm;
                    finishdir.X = 0.0; finishdir.Y = 0.0; finishdir.Z = -1.0; 
                    finishnorm.X = 0.0; finishnorm.Y = -1.0; finishnorm.Z = 0.0;
                    InterpolateCamera(startdir, startnorm, finishdir, finishnorm, 45 / trkSpeed.Value, 1.0, ar, "");

                    startdir = finishdir;
                    startnorm = finishnorm;
 */
                    finishdir.X = -0.1; finishdir.Y = 0; finishdir.Z = -1.0; finishdir = finishdir.UnitVector;
                    finishnorm.X = 0.0; finishnorm.Y = -1.0; finishnorm.Z = 0.0; finishnorm = (finishnorm - (finishnorm * finishdir) * finishdir).UnitVector;
                    InterpolateCamera(startdir, startnorm, finishdir, finishnorm, 90 / trkSpeed.Value, 1.0, ar, "");
                    for (i = 0; i <= (int)DataMap[4, 1]; i += trkSpeed.Value)
                    {
                        double theta = (Math.PI * 2.0 * i) / (int)DataMap[4, 1];
                        double cth = -Math.Cos(theta);
                        double sth = -Math.Sin(theta);
                        finishdir.X = cth * 0.1;
                        finishdir.Y = sth * 0.1;
                        finishdir.Z = -1.0;
                        finishdir = finishdir.UnitVector;
                        finishnorm.X = 0.0; finishnorm.Y = -1.0; finishnorm.Z = 0.0; finishnorm = (finishnorm - (finishnorm * finishdir) * finishdir).UnitVector;
                        ar.Add(new object[] { finishdir, finishnorm, finishdir ^ finishnorm, "XY pan" });
                    }
                    startdir = finishdir;
                    startnorm = finishnorm;
                }
                InterpolateCamera(startdir, startnorm, loopdir, loopnorm, 90 / trkSpeed.Value, 1.0, ar, "");

                pbGeneration.Maximum = ar.Count;
                pbGeneration.Value = 0;
                foreach (object [] o in ar)
                {
                    SySal.BasicTypes.Vector d = (SySal.BasicTypes.Vector)o[0];
                    SySal.BasicTypes.Vector n = (SySal.BasicTypes.Vector)o[1];
                    gdiDisplay.SetCameraOrientation(d.X, d.Y, d.Z, n.X, n.Y, n.Z);
                    gdiDisplay.SetCameraSpotting(PX, PY, PZ);
                    System.Drawing.Graphics g = System.Drawing.Graphics.FromImage(b);
                    gdiDisplay.Render(g);
                    int margin = 0;
                    if (chkEvent.Checked)
                    {
                        g.DrawString("Event " + m_Event, new Font("Verdana", 12), new SolidBrush( Color.White), 0, margin);
                        margin += 16;
                    }
                    if (chkBrick.Checked)
                    {
                        g.DrawString("Brick " + m_Brick, new Font("Verdana", 12), new SolidBrush(Color.White), 0, margin);
                        margin += 16;
                    }
                    string comment = (string)o[3];
                    if (chkAddViewExplanation.Checked && comment != null && comment.Length > 0)
                    {
                        g.DrawString(comment, new Font("Verdana", 12), new SolidBrush(Color.White), 0, margin);
                        margin += 16;
                    }
                    mv.AddFrame(b);
                    pbGeneration.Value = mv.Frames;
                    Application.DoEvents();
                }
                mv.Save(sdlg.FileName);
                Cursor = oldc;
                MessageBox.Show("File correctly saved", "Movie generation OK", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception x)
            {
                Cursor = oldc;
                MessageBox.Show("Error saving file\r\n" + x.ToString(), "Movie generation error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            gdiDisplay.SetCameraOrientation(DX, DY, DZ, NX, NY, NZ);
            gdiDisplay.SetCameraSpotting(PX, PY, PZ);
            gdiDisplay.AutoRender = true;
        }

        object[,] DataMap;

        void InitDataMap()
        {
            DataMap = new object[5, 2]
            {
                { txtXYFrames, 20 },
                { txtXZFrames, 20 },
                { txtYZFrames, 20 },
                { txtZRotFrames, 20 },
                { txtXYPanFrames, 20 }
            };
            int i;
            for (i = 0; i < DataMap.GetLength(0); i++)
                ((TextBox)DataMap[i, 0]).Text = DataMap[i, 1].ToString();
        }

        private void OnTextBoxLeave(object sender, EventArgs e)
        {
            int i;
            for (i = 0; i < DataMap.GetLength(0); i++)
                if (DataMap[i, 0] == sender)
                    try
                    {
                        DataMap[i, 1] = Convert.ToInt32(((TextBox)sender).Text);
                    }
                    catch (Exception)
                    {
                        ((TextBox)sender).Text = DataMap[i, 1].ToString();
                    }
        }

        SySal.BasicTypes.Vector[] InterpolateNormalVector(SySal.BasicTypes.Vector start, SySal.BasicTypes.Vector finish, int steps, double smoothing)
        {
            SySal.BasicTypes.Vector[] ret = new SySal.BasicTypes.Vector[steps];            
            start = start.UnitVector;
            finish = finish.UnitVector;

            SySal.BasicTypes.Vector j = (start ^ finish);
            SySal.BasicTypes.Vector k = new SySal.BasicTypes.Vector();
            if (j.Norm2 > 0.0) j = j.UnitVector;
            k = j ^ start;

            double cf = finish * start;
            if (cf > 1.0) cf = 1.0;
            else if (cf < -1.0) cf = -1.0;

            double theta = Math.Acos(cf);

            if (finish * k < 0) theta = -theta;

            int i;
            for (i = 0; i < steps; i++)
            {
                double lambda = (double)i / (double)(steps - 1);
                double phi = ((1.0 - smoothing) * lambda + smoothing * (1.0 - Math.Cos(Math.PI * lambda)) * 0.5) * theta;
                double cphi = Math.Cos(phi);
                double sphi = Math.Sin(phi);
                ret[i] = cphi * start + sphi * k;
            }
            
            return ret;
        }

        void InterpolateCamera(SySal.BasicTypes.Vector startdir, SySal.BasicTypes.Vector startnorm, SySal.BasicTypes.Vector finishdir, SySal.BasicTypes.Vector finishnorm, int steps, double smoothing, System.Collections.ArrayList ar, string comment)
        {
            SySal.BasicTypes.Vector d = new SySal.BasicTypes.Vector();
            SySal.BasicTypes.Vector n = new SySal.BasicTypes.Vector();
            SySal.BasicTypes.Vector b = new SySal.BasicTypes.Vector();
            int i;
            SySal.BasicTypes.Vector[] td = InterpolateNormalVector(startdir, finishdir, steps, smoothing);                            
            SySal.BasicTypes.Vector[] tn = InterpolateNormalVector(startnorm, finishnorm, steps, smoothing);
            for (i = 0; i < steps; i++)
            {
                d = td[i];
                n = (tn[i] - ((tn[i] * td[i]) * td[i])).UnitVector;
                b = td[i] ^ n;
                ar.Add(new object[4] { d, n, b, comment }); 
            }                       
        }

        private void OnTxtCommentChanged(object sender, EventArgs e)
        {

        }
    }
}