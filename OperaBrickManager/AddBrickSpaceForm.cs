using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace OperaBrickManager
{
    public partial class AddBrickSpaceForm : Form
    {
        public int NewID;
        public AddBrickSpaceForm()
        {
            InitializeComponent();
        }

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

        private void AddBrickSpaceForm_Load(object sender, EventArgs e)
        {
            txtBrickSet.Text = SetBrick;
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            try
            {
                ReturnData();
                DialogResult = DialogResult.OK;
                Close();
                
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "You need insert a number", MessageBoxButtons.OK, MessageBoxIcon.Error);
                txtNewID.Text = "";
                
                
            }
        }

        private void ReturnData()
        {
            NewID = System.Convert.ToInt32(txtNewID.Text);
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            Close();
        }
    }
}