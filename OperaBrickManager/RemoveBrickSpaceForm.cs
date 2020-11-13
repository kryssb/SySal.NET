using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace OperaBrickManager
{
    public partial class RemoveBrickSpaceForm : Form
    {
        public int OldIdSpace;
        public string SetBrick
        {
            get
            {
                return txtBrickSet.Text;

            }
            set
            {
                txtBrickSet.Text = value;
            }
        }
        public RemoveBrickSpaceForm()
        {
            InitializeComponent();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void btnOk_Click(object sender, EventArgs e)
        {
            try
            {
                Results();
                DialogResult = DialogResult.OK;
                Close();
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "You need insert a number", MessageBoxButtons.OK, MessageBoxIcon.Error);
                txtIdSpace.Text = "";
            }
        }

        private void Results()
        {
            OldIdSpace = System.Convert.ToInt32(txtIdSpace.Text);
            
 
        }

        private void RemoveBrickSpaceForm_Load(object sender, EventArgs e)
        {
            txtBrickSet.Text = SetBrick;
        }

        
    }
}