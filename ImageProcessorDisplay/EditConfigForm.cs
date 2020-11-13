using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.ImageProcessorDisplay
{
    public partial class EditConfigForm : Form
    {
        public EditConfigForm(SySal.ImageProcessorDisplay.Configuration c)
        {
            InitializeComponent();
            C = (Configuration)c.Clone();
        }

        internal SySal.ImageProcessorDisplay.Configuration C;

        private void OnLeftLeave(object sender, EventArgs e)
        {
            try
            {
                C.PanelLeft = Convert.ToInt32(txtLeft.Text);
            }
            catch (Exception) 
            {
                txtLeft.Text = C.PanelLeft.ToString();
                txtLeft.Focus();
            }
        }

        private void OnTopLeave(object sender, EventArgs e)
        {
            try
            {
                C.PanelTop = Convert.ToInt32(txtTop.Text);
            }
            catch (Exception)
            {
                txtTop.Text = C.PanelTop.ToString();
                txtTop.Focus();
            }
        }

        private void OnWidthLeave(object sender, EventArgs e)
        {
            try
            {
                C.PanelWidth = Convert.ToUInt16(txtWidth.Text);
            }
            catch (Exception)
            {
                txtWidth.Text = C.PanelWidth.ToString();
                txtWidth.Focus();
            }
        }

        private void OnHeightLeave(object sender, EventArgs e)
        {
            try
            {
                C.PanelHeight = Convert.ToUInt16(txtHeight.Text);
            }
            catch (Exception)
            {
                txtHeight.Text = C.PanelHeight.ToString();
                txtHeight.Focus();
            }
        }

        private void OnLoad(object sender, EventArgs e)
        {
            txtLeft.Text = C.PanelLeft.ToString();
            txtTop.Text = C.PanelTop.ToString();
            txtWidth.Text = C.PanelWidth.ToString();
            txtHeight.Text = C.PanelHeight.ToString();
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.OK;
            Close();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }        
    }
}