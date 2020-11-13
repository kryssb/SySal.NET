using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.DAQSystem.Drivers.PredictionScan3Driver
{
    public partial class EnqueueOpForm : Form
    {
        public struct TrackFollowInfo
        {
            public long Id;
            public SySal.BasicTypes.Vector2 Position;
            public SySal.BasicTypes.Vector2 Slope;            
            public int Plate;
        }

        public long BrickId;        

        int m_MaxZPlate;

        int m_MinZPlate;

        string m_AutoStartPath;

        public TrackFollowInfo[] Tracks = new TrackFollowInfo[0];

        static double sWidthHeight = 3000.0;

        static string sSearchString = "instr(upper(description), 'TRACK') >= 0 && instr(upper(description), 'FOLLOW') >= 0";

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
                if (chkFavorites.Checked) sql = "select to_number(value), substr(name, length('PROGSET') + 2) as description from opera.lz_sitevars where name like 'PROGSET %' and instr(upper(name), 'TRACK') > 0 and instr(upper(name), 'FOLLOW') > 0 and exists (select * from tb_programsettings where id = to_number(value) and driverlevel = 2) order by value";
                else sql = "select id, description from tb_programsettings where driverlevel = 2 and instr(upper(description), 'TRACK') >= 0 and instr(upper(description), 'FOLLOW') > 0 order by id";
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
                    lvi.Tag = /*rd1.GetInt64(0)*/ rd1.GetString(1);
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
                rd4 = new SySal.OperaDb.OperaDbCommand("select min(id), max(id) from tb_plates where id_eventbrick = " + BrickId, conn).ExecuteReader();
                if (rd4.Read() == false)
                {
                    MessageBox.Show("Wrong brick.\r\nCannot continue.", "Data Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    Close();
                    return;
                }
                m_MinZPlate = rd4.GetInt32(0);
                m_MaxZPlate = rd4.GetInt32(1);
                txtNotes.Text = "Track Follow on brick " + BrickId + ", tracks = " + Tracks.Length;
                RefreshProgramSettings(conn);
                OnWidthHeightLeave(this, null);
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
                if (rd4 != null) rd4.Close();
                if (conn != null) conn.Close();
            }
        }

        private void OnWidthHeightLeave(object sender, EventArgs e)
        {
            try
            {
                sWidthHeight = Convert.ToDouble(txtWidthHeight.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtWidthHeight.Text = sWidthHeight.ToString(System.Globalization.CultureInfo.InvariantCulture);
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
                string insertinfo = "\r\n" + lvProgramSettings.SelectedItems[0].Tag.ToString() + " $ " + lvMachines.SelectedItems[0].Tag.ToString() + " $ " + BrickId + " $ 0 $ " + txtNotes.Text.Replace('$','_') + " $ Volumes " + Tracks.Length;
                foreach (TrackFollowInfo tk in Tracks)
                {
                    insertinfo += "; " + tk.Id + " " +
                        (tk.Position.X - 0.5 * sWidthHeight).ToString(System.Globalization.CultureInfo.InvariantCulture) + " " + (tk.Position.X + 0.5 * sWidthHeight).ToString(System.Globalization.CultureInfo.InvariantCulture) + " " +
                        (tk.Position.Y - 0.5 * sWidthHeight).ToString(System.Globalization.CultureInfo.InvariantCulture) + " " + (tk.Position.Y + 0.5 * sWidthHeight).ToString(System.Globalization.CultureInfo.InvariantCulture) + " " +
                        m_MinZPlate + " " + m_MaxZPlate + " " + tk.Plate + " " + tk.Slope.X.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) + " " + tk.Slope.Y.ToString("F5", System.Globalization.CultureInfo.InvariantCulture);
                }
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
    }
}