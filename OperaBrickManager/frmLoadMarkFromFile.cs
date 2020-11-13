using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.IO;

namespace OperaBrickManager
{
    public partial class frmLoadMarkFromFile : Form
    {
        public frmLoadMarkFromFile()
        {
            InitializeComponent();
        }


        public string NumberBrick
        {
            get
            {
                return txtIdBrick.Text;

            }
            set
            {
                txtIdBrick.Text = value;

            }
        }

        public class CoordinateMark
        {
            public SySal.BasicTypes.Vector[] Marks;
            public char type;            
        };

        public CoordinateMark MarkSet = new CoordinateMark();

        private void btnLoadFromFile_Click(object sender, EventArgs e)
        {
            char flag;
            if (FileOpen.ShowDialog() == DialogResult.OK)
            {
                flag = Checked();

                if (flag == 'L' || flag == 'S' || flag == 'X')
                {
                    GetPositions(FileOpen.FileName, flag);                  
                }                                              
            }
        }

        private char Checked()
        {
            char type;
            if (rdbCSMark.Checked == true && rdbLateralMark.Checked == false && rdbSpotMark.Checked == false)
            {
                type = 'X';
            }
             else if (rdbCSMark.Checked == false && rdbLateralMark.Checked == true && rdbSpotMark.Checked == false)
            {
                type = 'L';
            }
            else if (rdbCSMark.Checked == false && rdbLateralMark.Checked == false && rdbSpotMark.Checked == true)
            {
                type = 'S';
            }
            else
            {
                MessageBox.Show("You must select a set of mark");
                return 'N';
            }
            return type;

        }

        private void GetPositions(String NameFile, char typeMark)
        {
            try
            {

                char[] sep1 = { ':' };
                char[] sep2 = { ';' };
                char[] sep3 = { ' ' };
                String txtBuffer;
                int counter = 0;
                int NumberMark;
                
                System.Text.RegularExpressions.Regex marktype_rx = new System.Text.RegularExpressions.Regex(@"\s*(\w+):\s*\d+\s+\d+\s+\d+\s+\d+\s*[;\n]");
                System.Text.RegularExpressions.Regex count_rx = new System.Text.RegularExpressions.Regex(@"\s*(\d+)\s+\S+\s+\S+\s+\S+\s+\S+\s*[;\n]");
                System.Text.RegularExpressions.Regex mark_rx = new System.Text.RegularExpressions.Regex(@"\s*\d+\s+(\S+)\s+(\S+)\s+\S+\s+\S+\s+\d+\s+\d+\s+(\d+)\s*");

                StreamReader txtStream = new StreamReader(NameFile);
                txtBuffer = txtStream.ReadToEnd();
                txtStream.Close();
                System.Text.RegularExpressions.Match mh = marktype_rx.Match(txtBuffer);
                if (mh.Success == false) throw new Exception("Wrong syntax in string beginning");
                if (mh.Groups[1].Value.ToLower() != "mapx" && mh.Groups[1].Value.ToLower() != "mapext") throw new Exception("Unknown mark set type. Known types are \"mapX\" and \"mapext\".");
                System.Text.RegularExpressions.Match mn = count_rx.Match(txtBuffer, mh.Index + mh.Length + 1);
                if (mn.Success == false) throw new Exception("Can't read number of marks");
                MarkSet.Marks = new SySal.BasicTypes.Vector[System.Convert.ToInt32(mn.Groups[1].Value)];
                int pos = mn.Index + mn.Length + 1;
                int marknum;
                for (marknum = 0; marknum < MarkSet.Marks.Length; marknum++)
                {
                    System.Text.RegularExpressions.Match mk = mark_rx.Match(txtBuffer, pos);
                    if (mk.Success == false) throw new Exception("Error reading mark number " + marknum + ".");
                    pos = mk.Index + mk.Length + 1;
                    MarkSet.Marks[marknum].X = System.Convert.ToDouble(mk.Groups[1].Value);
                    MarkSet.Marks[marknum].Y = System.Convert.ToDouble(mk.Groups[2].Value);
                    MarkSet.Marks[marknum].Z = System.Convert.ToDouble(mk.Groups[3].Value);
                }
                MarkSet.type = typeMark;
                this.txtNcount.Text = MarkSet.Marks.Length.ToString();              
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Input format string is not valid", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

        }

        private void frmLoadMarkFromFile_Load(object sender, EventArgs e)
        {
            txtIdBrick.Text = NumberBrick;
        }

        private void noChange(object sender, EventArgs e)
        {
            txtIdBrick.ReadOnly = true;
        }

        private void noChange2(object sender, EventArgs e)
        {
            txtNcount.ReadOnly = true;
            btnOk.Enabled = true;
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void btnOk_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.OK;
            Close();
        }

        private void btnClear_Click(object sender, EventArgs e)
        {
            rdbCSMark.Checked = false;
            rdbLateralMark.Checked = false;
            rdbSpotMark.Checked = false;
            txtNcount.ReadOnly = false;
            txtNcount.Text = "";
            txtNcount.ReadOnly = true;
            btnOk.Enabled = false;
        }
       


       
    }
}