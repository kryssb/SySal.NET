using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.EasyReconstruct
{
    /// <summary>
    /// Provides a simple interface to launch TotalScan jobs.
    /// </summary>
    public partial class EnqueueOpForm : Form
    {
        /// <summary>
        /// Relevant information for a TrackFollow or TotalScan job.
        /// </summary>
        public struct TrackFollowInfo
        {
            /// <summary>
            /// Id of the volume to be created.
            /// </summary>
            public long Id;
            /// <summary>
            /// Position of the center of the volume at the reference plate.
            /// </summary>
            public SySal.BasicTypes.Vector2 Position;
            /// <summary>
            /// Skewing slopes for the volume.
            /// </summary>
            public SySal.BasicTypes.Vector2 Slope;            
            /// <summary>
            /// The reference slope.
            /// </summary>
            public int Plate;
        }
        /// <summary>
        /// The ID of the brick on which the scanning has to be performed.
        /// </summary>
        public long BrickId;        

        string m_AutoStartPath;
        /// <summary>
        /// The VolumeStart information.
        /// </summary>
        public TrackFollowInfo VolStart;

        /// <summary>
        /// Set to <c>true</c> if the volume starts from a track, <c>false</c> otherwise.
        /// </summary>
        public bool VolumeStartsFromTrack;

        /// <summary>
        /// Creates a new form.
        /// </summary>
        public EnqueueOpForm()
        {
            InitializeComponent();
        }

        void RefreshProgramSettings(SySal.OperaDb.OperaDbConnection conn)
        {
            SySal.OperaDb.OperaDbDataReader rd2 = null;
            try
            {
                lvProgramSettings.Items.Clear();
                string sql = "";
                if (chkFavorites.Checked) sql = "select to_number(value), substr(name, length('PROGSET') + 2) as description from opera.lz_sitevars where name like 'PROGSET %' and exists (select * from tb_programsettings where id = to_number(value) and driverlevel = 2 and instr(upper(executable), 'TOTALSCAN') > 0) order by value";
                else sql = "select id, description from tb_programsettings where driverlevel = 2 and instr(upper(executable), 'TOTALSCAN') > 0 order by id";
                rd2 = new SySal.OperaDb.OperaDbCommand(sql, conn).ExecuteReader();
                while (rd2.Read())
                {
                    ListViewItem lvi = new ListViewItem(rd2.GetInt64(0).ToString());
                    lvi.Tag = rd2.GetInt64(0);
                    lvi.SubItems.Add(rd2.GetString(1));
                    lvProgramSettings.Items.Add(lvi);
                }
            }
            catch (Exception x) 
            {
                MessageBox.Show(x.ToString(), "Error Accessing DB", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (rd2 != null) rd2.Close();
            }
        }

        struct PlateRecord
        {
            public int Id;
            public double Z;
            public PlateRecord(int id, double z) { Id = id; Z = z; }
        }

        PlateRecord[] Plates;

        private void OnLoad(object sender, EventArgs e)
        {
            SySal.OperaDb.OperaDbConnection conn = null;
            SySal.OperaDb.OperaDbDataReader rd1 = null;
            SySal.OperaDb.OperaDbDataReader rd3 = null;
            SySal.OperaDb.OperaDbDataReader rd4 = null; 
            try
            {
                conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
                conn.Open();
                rd1 = new SySal.OperaDb.OperaDbCommand("select id, name from tb_machines where id_site = (select value from opera.lz_sitevars where name = 'ID_SITE') and isscanningserver > 0 order by name", conn).ExecuteReader();
                while (rd1.Read())
                {
                    ListViewItem lvi = new ListViewItem(rd1.GetInt64(0).ToString());
                    lvi.Tag = rd1.GetString(1) /*rd1.GetInt64(0)*/;
                    lvi.SubItems.Add(rd1.GetString(1));
                    lvMachines.Items.Add(lvi);
                }
                rd1.Close();
                rd3 = new SySal.OperaDb.OperaDbCommand("select value from opera.lz_sitevars where name = 'BM_AutoStartFile'", conn).ExecuteReader();
                if (rd3.Read()) m_AutoStartPath = rd3.GetString(0);
                else
                {
                    MessageBox.Show("AutoStart file path not set in LZ_SITEVARS.\r\nCannot continue.", "Infrastructure Setup Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    Close();
                    return;
                }
                System.Collections.ArrayList pa = new System.Collections.ArrayList();
                rd4 = new SySal.OperaDb.OperaDbCommand("select id, z from tb_plates where id_eventbrick = " + BrickId + " order by z desc", conn).ExecuteReader();
                while (rd4.Read())                
                    pa.Add(new PlateRecord(rd4.GetInt32(0), rd4.GetDouble(1)));
                if (pa.Count == 0) throw new Exception("No plates registered in your DB for this brick.\r\nPlease check and retry.");
                Plates = (PlateRecord[])pa.ToArray(typeof(PlateRecord));
                int i;
                for (i = 0; i < Plates.Length && Plates[i].Id != VolStart.Plate; i++) ;
                cmbPlates.Items.Add(i + "/0");
                cmbPlates.Items.Add("0/" + (Plates.Length - 1 - i));
                cmbSkew.Items.Add(VolStart.Slope.X.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "/" + VolStart.Slope.Y.ToString("F4", System.Globalization.CultureInfo.InvariantCulture));
                rd4.Close();
                cmbNotes.Items.Add("TotalScan on brick " + BrickId);
                cmbNotes.Items.Add("TotalScan on brick " + BrickId + " around vertex point");
                cmbNotes.Items.Add("TotalScan on brick " + BrickId + " around stopping point");
                cmbNotes.SelectedIndex = VolumeStartsFromTrack ? 2 : 1;
                RefreshProgramSettings(conn);
                cmbWidthHeight.SelectedIndex = 0;
                cmbPlates.SelectedIndex = 0;
                cmbSkew.SelectedIndex = 0;
                OnWidthHeightLeave(this, null);
                OnPlatesLeave(this, null);
                OnSkewLeave(this, null);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Initialization Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Close();
            }
            finally
            {
                if (rd1 != null) rd1.Close();
                if (rd3 != null) rd3.Close();
                if (rd4 != null) rd3.Close();  
                if (conn != null) conn.Close();
            }
        }

        double sWidth;

        double sHeight;

        private void OnWidthHeightLeave(object sender, EventArgs e)
        {
            try
            {
                string[] t = cmbWidthHeight.Text.Split('/');
                if (t.Length != 2) throw new Exception();
                sWidth = Convert.ToDouble(t[0], System.Globalization.CultureInfo.InvariantCulture);
                sHeight = Convert.ToDouble(t[1], System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                cmbWidthHeight.Text = sWidth.ToString(System.Globalization.CultureInfo.InvariantCulture) + "/" + sHeight.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            if (lvMachines.SelectedItems.Count == 1 && lvProgramSettings.SelectedItems.Count == 1)
            {
                string insertinfo = "\r\n" + lvProgramSettings.SelectedItems[0].Tag.ToString() + " $ " + lvMachines.SelectedItems[0].Tag.ToString() + " $ " + BrickId + " $ 0 $ " + cmbNotes.Text.Replace('$', '_') + " $ Volumes 1";
                int i;
                for (i = 0; i < Plates.Length && Plates[i].Id != VolStart.Plate; i++) ;
                int dwid, upid;
                dwid = Plates[Math.Max(0, i - sDownPlates)].Id;
                upid = Plates[Math.Min(Plates.Length - 1, i + sUpPlates)].Id;
                insertinfo += "; " + VolStart.Id + " " +
                        (VolStart.Position.X - 0.5 * sWidth).ToString(System.Globalization.CultureInfo.InvariantCulture) + " " + (VolStart.Position.X + 0.5 * sWidth).ToString(System.Globalization.CultureInfo.InvariantCulture) + " " +
                        (VolStart.Position.Y - 0.5 * sHeight).ToString(System.Globalization.CultureInfo.InvariantCulture) + " " + (VolStart.Position.Y + 0.5 * sHeight).ToString(System.Globalization.CultureInfo.InvariantCulture) + " " +
                        upid + " " + dwid + " " + VolStart.Plate + " " + sSkew.X.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + " " + sSkew.Y.ToString("F5", System.Globalization.CultureInfo.InvariantCulture);
                try
                {
                    SaveFileDialog sdlg = new SaveFileDialog();
                    sdlg.Title = "Select AutoStart file";
                    sdlg.Filter = "Text files (*.txt)|*.txt";
                    if (sdlg.ShowDialog() == DialogResult.OK)
                    {
                        System.IO.File.AppendAllText(sdlg.FileName, insertinfo);
                        MessageBox.Show("AutoStart updated", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                        Close();
                    }
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.ToString(), "File Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private void OnFavoriteChanged(object sender, EventArgs e)
        {
            SySal.OperaDb.OperaDbConnection conn = null;
            try
            {
                conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
                conn.Open();
                RefreshProgramSettings(conn);
            }
            catch (Exception) { }
            finally
            {
                if (conn != null) conn.Close();
            }
        }

        uint sDownPlates;

        uint sUpPlates;

        private void OnPlatesLeave(object sender, EventArgs e)
        {
            try
            {
                string[] t = cmbPlates.Text.Split('/');
                if (t.Length != 2) throw new Exception();
                sDownPlates = Convert.ToUInt32(t[0], System.Globalization.CultureInfo.InvariantCulture);
                sUpPlates = Convert.ToUInt32(t[1], System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                cmbPlates.Text = sDownPlates.ToString(System.Globalization.CultureInfo.InvariantCulture) + "/" + sUpPlates.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        SySal.BasicTypes.Vector2 sSkew;

        private void OnSkewLeave(object sender, EventArgs e)
        {
            try
            {
                string[] t = cmbSkew.Text.Split('/');
                if (t.Length != 2) throw new Exception();
                sSkew.X = Convert.ToDouble(t[0], System.Globalization.CultureInfo.InvariantCulture);
                sSkew.Y = Convert.ToDouble(t[1], System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                cmbSkew.Text = sSkew.X.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "/" + sSkew.Y.ToString("F4", System.Globalization.CultureInfo.InvariantCulture);
            }
        }
    }
}