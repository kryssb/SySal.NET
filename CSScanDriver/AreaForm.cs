using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.DAQSystem.Drivers.CSScanDriver
{
    public partial class AreaForm : Form
    {
        public string InterruptString = null;

        public AreaForm()
        {
            InitializeComponent();
        }

        System.Text.RegularExpressions.Regex ZoneEx = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*");

        private bool IsFormValid()
        {
            try
            {
                if (textMinX.Text == null || textMinX.Text.Trim() == "") return false;
                if (textMaxX.Text == null || textMaxX.Text.Trim() == "") return false;
                if (textMinY.Text == null || textMinY.Text.Trim() == "") return false;
                if (textMaxY.Text == null || textMaxY.Text.Trim() == "") return false;

                MinX = Convert.ToDouble(textMinX.Text, System.Globalization.CultureInfo.InvariantCulture);
                MaxX = Convert.ToDouble(textMaxX.Text, System.Globalization.CultureInfo.InvariantCulture);
                MinY = Convert.ToDouble(textMinY.Text, System.Globalization.CultureInfo.InvariantCulture);
                MaxY = Convert.ToDouble(textMaxY.Text, System.Globalization.CultureInfo.InvariantCulture);
 
                if ((MinX > MaxX) || (MinY > MaxY)) throw new Exception("area not valid");

                InterruptString = Convert.ToString("Zone " + MinX.ToString() + " " + MaxX.ToString() + " " + MinY.ToString() + " " + MaxY.ToString());
                System.Text.RegularExpressions.Match mz = ZoneEx.Match(InterruptString);
                if (mz.Success == false)
                {
                    InterruptString = null;
                    MessageBox.Show("area string not valid");
                    return false;
                }
            }
            catch (Exception x)
            {
                InterruptString = null;
                MessageBox.Show(x.Message, "Error");
                return false;
            }

            return true;
        }

        double MinX, MaxX, MinY, MaxY;

        private void buttonOk_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void buttonCancel_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void textMinX_TextChanged(object sender, EventArgs e)
        {
            buttonOk.Enabled = IsFormValid();
        }

        private void textMaxX_TextChanged(object sender, EventArgs e)
        {
            buttonOk.Enabled = IsFormValid();
        }

        private void textMinY_TextChanged(object sender, EventArgs e)
        {
            buttonOk.Enabled = IsFormValid();
        }

        private void textMaxY_TextChanged(object sender, EventArgs e)
        {
            buttonOk.Enabled = IsFormValid();
        }
    }
}