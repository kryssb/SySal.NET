using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Processing.TagPrimary
{
    public partial class EditConfigForm : Form
    {
        public EditConfigForm()
        {
            InitializeComponent();
        }

        public SySal.Processing.TagPrimary.Configuration C;

        private bool ValidateDouble(TextBox txt, ref double v)
        {
            try
            {
                v = System.Convert.ToDouble(txt.Text, System.Globalization.CultureInfo.InvariantCulture);
                return true;
            }
            catch (Exception)
            {
                txt.Text = v.ToString(System.Globalization.CultureInfo.InvariantCulture);
                txt.Focus();
                return false;
            }
        }

        private bool ValidateInt(TextBox txt, ref int v)
        {
            try
            {                
                v = System.Convert.ToInt32(txt.Text);                
                return true;
            }
            catch (Exception)
            {
                txt.Text = v.ToString();
                txt.Focus();
                return false;
            }
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            if (ValidateAll(true))
            {
                DialogResult = DialogResult.OK;
                Close();
            }
        }

        private bool ValidateAll(bool stoponerror)
        {
            if (ValidateDouble(txtScanBackPosTol,   ref C.PositionToleranceSB) == false && stoponerror) return false;
            if (ValidateDouble(txtScanBackSlopeTol, ref C.AngularToleranceSB)  == false && stoponerror) return false;
            if (ValidateDouble(txtCSPosTol,         ref C.PositionToleranceCS) == false && stoponerror) return false;
            if (ValidateDouble(txtCSSlopeTol,       ref C.AngularToleranceCS)  == false && stoponerror) return false;
            if (ValidateDouble(txtEleDetPosTol,     ref C.EledetPosTol)        == false && stoponerror) return false;
            if (ValidateDouble(txtEleDetSlopeTol,   ref C.EledetAngTol)        == false && stoponerror) return false;
            return true;
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }
    }
}