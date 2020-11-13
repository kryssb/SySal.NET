using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Text.RegularExpressions;


namespace OperaBrickManager
{
    
    public partial class frmAddBrick : Form
    {
        public frmAddBrick()
        {
            InitializeComponent();
        }
        public int Brick;
        public int NPlate;
        public int DPlate;
        public double ZMax;
        public double GLMinX;
        public double GlMinY;
        public double GlMaxX;
        public double GlMaxY;
        public double ZeroX, ZeroY;
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
        private void btnInputDataFiles_Click(object sender, EventArgs e)
        {
            if (dlgOpenFile.ShowDialog() == DialogResult.OK)
            GetCoordinate(dlgOpenFile.FileName, true);
        }
        private void GetCoordinate (String NameFile, bool UseAll)
        {
            try
            {
                char[] sep1 = { ':' };
                char[] sep2 = { ';' };
                char[] sep3 = { ' ' };
                String txtBuffer;
                StreamReader txtStream = new StreamReader(NameFile);
                txtBuffer = txtStream.ReadLine();
                string[] Data = txtBuffer.Split(sep1);
                Data = Data[1].Split(sep2);
                Data = Data[1].Split(sep3);
                if (UseAll)
                {
                    this.txtEmuMinX.Text = Data[2];
                    this.txtEmuMinY.Text = Data[3];
                    this.txtEmuMaxX.Text = Data[4];
                    this.txtEmuMaxY.Text = Data[5];
                }
                else
                {
                    this.txtZeroX.Text = Data[2];
                    this.txtZeroY.Text = Data[3];
                }
                return;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Invalid input format", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
        private void btnLoadZeroCoordinate_Click(object sender, EventArgs e)
        {
         if (dlgOpenFile.ShowDialog() == DialogResult.OK)
             GetCoordinate(dlgOpenFile.FileName, false);
        }
        private void Filled1(object sender, EventArgs e)
        {
            btnInputDataFiles.Enabled =!((this.txtEmuMinX.Text.Length +
                    this.txtEmuMinY.Text.Length +
                    this.txtEmuMaxX.Text.Length +
                    this.txtEmuMaxY.Text.Length) > 0);
            if (btnInputDataFiles.Enabled || ((this.txtEmuMinX.Text.Length *
                                                 this.txtEmuMinY.Text.Length *
                                                 this.txtEmuMaxX.Text.Length *
                                                 this.txtEmuMaxY.Text.Length) != 0)

                 ) 
                grpXY.Tag = "1";
            else 
                grpXY.Tag = "0";

        }
        private void Filled2(object sender, EventArgs e)
        {
            btnLoadZeroCoordinate.Enabled = !((this.txtZeroX.Text.Length +
                                               this.txtZeroY.Text.Length) > 0);
            btnDBLink.Enabled = btnLoadZeroCoordinate.Enabled;
            if (btnLoadZeroCoordinate.Enabled || ((this.txtZeroX.Text.Length *
                                                this.txtZeroY.Text.Length)) != 0

                 )
                grpZeroXY.Tag = "1";
            else
                grpZeroXY.Tag = "0";
        }
        private void btnCompute_Click(object sender, EventArgs e)
        {
            try
            {
                double temp1, temp2;
                if (System.Convert.ToInt32(grpXY.Tag) * System.Convert.ToInt32(grpZeroXY.Tag) == 0)
                {
                    MessageBox.Show("Values must be input manually or loaded from file!", "Input warning");
                    return;
                }
                temp1 = System.Convert.ToDouble(txtEmuMinX.Text);
                temp2 = System.Convert.ToDouble(txtZeroX.Text);
                this.txtMinX.Text = System.Convert.ToString(temp1 + temp2);
                temp1 = System.Convert.ToDouble(txtEmuMaxX.Text);
                this.txtMaxX.Text = System.Convert.ToString(temp1 + temp2);
                temp1 = System.Convert.ToDouble(txtEmuMinY.Text);
                temp2 = System.Convert.ToDouble(txtZeroY.Text);
                this.txtMinY.Text = System.Convert.ToString(temp1 + temp2);
                temp1 = System.Convert.ToDouble(txtEmuMaxY.Text);
                this.txtMaxY.Text = System.Convert.ToString(temp1 + temp2);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Error in inserted value", MessageBoxButtons.OK);
            }
        }
        private void btnClearAll_Click(object sender, EventArgs e)
        {
           this.txtEmuMinX.Text = "";
           this.txtEmuMinY.Text ="";                                           
           this.txtEmuMaxX.Text = "";
           this.txtEmuMaxY.Text = "";                              
           this.txtZeroX.Text = "";
           this.txtZeroY.Text = "";
           grpXY.Tag = "0";
           grpZeroXY.Tag = "0";
        }
        private void btnOk_Click(object sender, EventArgs e)
        {
             try
             {
                
                ReturnData();
                DialogResult = DialogResult.OK;
                Close();
             }
            catch (Exception x)
             {
                MessageBox.Show(x.Message, "Not all fields are filled or error in a field", MessageBoxButtons.OK);
            }
            
        }
        private void btnCancel_Click(object sender, EventArgs e)
        {
            Close();
        }
        private void btnDBLink_Click(object sender, EventArgs e)
        {
            frmAddFromDb fAddFromDb = new frmAddFromDb();
            if (fAddFromDb.ShowDialog() == DialogResult.OK)
            {
                this.txtZeroX.Text = fAddFromDb.m_Result1;
                this.txtZeroY.Text = fAddFromDb.m_Result2; 
            }
            fAddFromDb.Dispose();
        }
        private void ReturnData()
        {
           
            Brick = System.Convert.ToInt32(this.txtBrick.Text);
            NPlate = System.Convert.ToInt32(this.cmbNumberOfPlates.Text);
            DPlate = System.Convert.ToInt32(this.cmbDownstreamPlate.Text);
            ZMax = System.Convert.ToDouble(this.txtMaxZ.Text);
            GLMinX = System.Convert.ToDouble(this.txtMinX.Text);
            GlMaxX = System.Convert.ToDouble(this.txtMaxX.Text);
            GlMinY = System.Convert.ToDouble(this.txtMinY.Text);
            GlMaxY = System.Convert.ToDouble(this.txtMaxY.Text);
            ZeroX = System.Convert.ToDouble(this.txtZeroX.Text);
            ZeroY = System.Convert.ToDouble(this.txtZeroY.Text);
        }
        private void frmAddBrick_Load(object sender, EventArgs e)
        {

            txtBrickSet.Text = SetBrick;
            
        }
        private void NoChange(object sender, EventArgs e)
        {
           
            txtBrickSet.ReadOnly = true;
        }
    }
    
}