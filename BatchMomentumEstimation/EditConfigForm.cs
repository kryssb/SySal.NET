using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using SySal.TotalScan;

namespace SySal.Processing.MCSLikelihood
{
    public partial class EditConfigForm : Form
    {
        public EditConfigForm()
        {
            InitializeComponent();
        }

        private void btnDefaultOPERA_Click(object sender, EventArgs e)
        {
            C.Geometry.Layers = new Geometry.LayerStart[116];
            ValidateLong(txtBrick, ref m_Brick, 0, 100000000);
            int i;
            for (i = 0; i < C.Geometry.Layers.Length; i += 2)
            {
                C.Geometry.Layers[i] = new Geometry.LayerStart();
                C.Geometry.Layers[i].RadiationLength = 29000.0;
                C.Geometry.Layers[i].ZMin = (i / 2 - 57) * 1300.0;
                C.Geometry.Layers[i].Brick = System.Convert.ToInt64(txtBrick.Text);
                C.Geometry.Layers[i].Plate = i / 2;
                C.Geometry.Layers[i + 1] = new Geometry.LayerStart();
                C.Geometry.Layers[i + 1].RadiationLength = 5600.0;
                C.Geometry.Layers[i + 1].ZMin = (i / 2 - 57) * 1300.0 + 300.0;
                C.Geometry.Layers[i + 1].Brick = System.Convert.ToInt64(txtBrick.Text);
                C.Geometry.Layers[i + 1].Plate = 0;
            }
            SetGeometryList();
        }

        static bool ValidateDouble(System.Windows.Forms.TextBox tb, ref double v, double minv, double maxv)
        {
            try
            {
                double a = System.Convert.ToDouble(tb.Text, System.Globalization.CultureInfo.InvariantCulture);
                if (a < minv || a > maxv) throw new Exception();
                v = a;
                return true;
            }
            catch (Exception)
            {
                tb.Text = v.ToString(System.Globalization.CultureInfo.InvariantCulture);
                tb.Focus();
                return false;
            }
        }

        static bool ValidateInt(System.Windows.Forms.TextBox tb, ref int v, double minv, double maxv)
        {
            try
            {
                int a = System.Convert.ToInt32(tb.Text, System.Globalization.CultureInfo.InvariantCulture);
                if (a < minv || a > maxv) throw new Exception();
                v = a;
                return true;
            }
            catch (Exception)
            {
                tb.Text = v.ToString(System.Globalization.CultureInfo.InvariantCulture);
                tb.Focus();
                return false;
            }
        }

        static bool ValidateLong(System.Windows.Forms.TextBox tb, ref long v, double minv, double maxv)
        {
            try
            {
                long a = System.Convert.ToInt64(tb.Text, System.Globalization.CultureInfo.InvariantCulture);
                if (a < minv || a > maxv) throw new Exception();
                v = a;
                return true;
            }
            catch (Exception)
            {
                tb.Text = v.ToString(System.Globalization.CultureInfo.InvariantCulture);
                tb.Focus();
                return false;
            }
        }

        public SySal.Processing.MCSLikelihood.Configuration C;

        private void OnSlopeErrLeave(object sender, EventArgs e)
        {
            ValidateDouble(txtSlopeErrors, ref C.SlopeError, 0.0, 0.100);
        }

        private void OnCLLeave(object sender, EventArgs e)
        {
            ValidateDouble(txtCL, ref C.ConfidenceLevel, 0.5, 1.0);
        }

        private void OnMinPLeave(object sender, EventArgs e)
        {
            ValidateDouble(txtPMin, ref C.MinimumMomentum, 0, 10000.0);
        }

        private void OnMaxPLeave(object sender, EventArgs e)
        {
            ValidateDouble(txtPMax, ref C.MaximumMomentum, 0, 10000.0);
        }

        private void OnPStepLeave(object sender, EventArgs e)
        {
            ValidateDouble(txtPStep, ref C.MomentumStep, 0.001, 10.0);
        }

        private void OnMinRadLenLeave(object sender, EventArgs e)
        {
            ValidateDouble(txtMinRadLen, ref C.MinimumRadiationLengths, 0, 100.0);
        }

        private void SetGeometryList()
        {
            lvGeometry.Items.Clear();
            foreach (Geometry.LayerStart ls in C.Geometry.Layers)
            {
                ListViewItem lvi = lvGeometry.Items.Add(ls.ZMin.ToString(System.Globalization.CultureInfo.InvariantCulture));
                lvi.SubItems.Add(ls.RadiationLength.ToString(System.Globalization.CultureInfo.InvariantCulture));
                lvi.SubItems.Add(ls.Plate.ToString());
                lvi.SubItems.Add(ls.Brick.ToString());
            }
        }

        private void EditConfigForm_Load(object sender, EventArgs e)
        {
            ValidateDouble(txtSlopeErrors, ref C.SlopeError, 0, 0);
            ValidateDouble(txtCL, ref C.ConfidenceLevel, 0, 0);
            ValidateDouble(txtPMin, ref C.MinimumMomentum, 0, 0);
            ValidateDouble(txtPMax, ref C.MaximumMomentum, 0, 0);
            ValidateDouble(txtPStep, ref C.MomentumStep, 0, 0);
            ValidateDouble(txtMinRadLen, ref C.MinimumRadiationLengths, 0, 0);
            ValidateInt(txtPlate, ref m_Plate, 0, 1000);
            ValidateLong(txtBrick, ref m_Brick, 0, 100000000);
            SetGeometryList();
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            if (C.MaximumMomentum < C.MinimumMomentum)
            {
                MessageBox.Show("Maximum momentum should be greater or equal to minimum momentum.", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            DialogResult = DialogResult.OK;
            Close();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        private void btnGeometryAdd_Click(object sender, EventArgs e)
        {
            Geometry.LayerStart ls = new Geometry.LayerStart();
            if (ValidateDouble(txtZMin, ref ls.ZMin, -1000000.0, 1000000.0) && ValidateDouble(txtRadLen, ref ls.RadiationLength, 0.0, 1000000.0) && ValidateInt(txtPlate, ref ls.Plate, 0, 1000) && ValidateLong(txtBrick, ref ls.Brick, 0, 100000000))
            {
                System.Collections.ArrayList ar = new System.Collections.ArrayList();
                ar.AddRange(C.Geometry.Layers);
                ar.Add(ls);
                ar.Sort(new Geometry.order());
                C.Geometry.Layers = (Geometry.LayerStart[])ar.ToArray(typeof(Geometry.LayerStart));
                SetGeometryList();
            }
        }

        private void btnGeometryDel_Click(object sender, EventArgs e)
        {
            if (lvGeometry.SelectedIndices.Count > 0)
            {
                int i;
                System.Collections.ArrayList ar = new System.Collections.ArrayList();
                ar.AddRange(C.Geometry.Layers);
                for (i = lvGeometry.SelectedIndices.Count - 1; i >= 0; i--)
                    ar.RemoveAt(lvGeometry.SelectedIndices[i]);
                ar.Sort(new Geometry.order());
                C.Geometry.Layers = (Geometry.LayerStart[])ar.ToArray(typeof(Geometry.LayerStart));
                SetGeometryList();
            }
        }

        private void btnDefaultStack_Click(object sender, EventArgs e)
        {
            C.Geometry.Layers = new Geometry.LayerStart[12];
            ValidateLong(txtBrick, ref m_Brick, 0, 100000000);
            int i;
            for (i = 1; i <= 10; i ++)
            {
                C.Geometry.Layers[i] = new Geometry.LayerStart();
                C.Geometry.Layers[i].RadiationLength = 29000.0;
                C.Geometry.Layers[i].ZMin = (i - 10) * 300;
                C.Geometry.Layers[i].Plate = i + 1;
                C.Geometry.Layers[i].Brick = m_Brick;
            }
            C.Geometry.Layers[0] = new Geometry.LayerStart();
            C.Geometry.Layers[0].RadiationLength = 1000000.0;
            C.Geometry.Layers[0].ZMin = -3000.0;
            C.Geometry.Layers[0].Plate = 1;
            C.Geometry.Layers[0].Brick = 0;
            C.Geometry.Layers[11] = new Geometry.LayerStart();
            C.Geometry.Layers[11].RadiationLength = 1000000.0;
            C.Geometry.Layers[11].ZMin = 300.0;
            C.Geometry.Layers[11].Plate = 12;
            C.Geometry.Layers[11].Brick = m_Brick;
            SetGeometryList();
        }

        private int m_Plate;

        private long m_Brick;

        private void btnFromDB_Click(object sender, EventArgs e)
        {
            if (ValidateLong(txtBrick, ref m_Brick, 0, 100000000) == false)
            {
                MessageBox.Show("A valid brick number is required.", "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            SySal.OperaDb.OperaDbConnection conn = null;
            try
            {
                conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
                conn.Open();
                SySal.OperaDb.Schema.DB = conn;
                SySal.OperaDb.Schema.TB_PLATES plates = SySal.OperaDb.Schema.TB_PLATES.SelectWhere("ID_EVENTBRICK = " + m_Brick, "Z ASC");
                Geometry.LayerStart[] ls = new Geometry.LayerStart[plates.Count * 2];
                int i;
                for (i = 0; i < plates.Count; i++)
                {
                    plates.Row = i;
                    ls[i * 2].Brick = m_Brick;
                    ls[i * 2].Plate = (int)plates._ID;
                    ls[i * 2].ZMin = plates._Z - 255.0;
                    ls[i * 2].RadiationLength = 29000.0;
                    ls[i * 2 + 1].Brick = m_Brick;
                    ls[i * 2 + 1].Plate = 0;
                    ls[i * 2 + 1].ZMin = plates._Z + 45.0;
                    ls[i * 2 + 1].RadiationLength = 5600.0;
                }
                ls[i * 2 - 1].RadiationLength = 1e9;
                C.Geometry.Layers = ls;
                SetGeometryList();
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "DB Connection Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (conn != null)
                {
                    conn.Close();
                    conn = null;
                }
                try
                {
                    SySal.OperaDb.Schema.DB = null;
                }
                catch (Exception) { }
            }
        }
   }
}