using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.SySal2GIF
{
    public partial class FilterControlForm : Form
    {
        public FilterControlForm()
        {
            InitializeComponent();
            DataMap = new object[4, 2]
            {
                {txtFilterMult, -0.25},
                {txtFilterZeroThreshold, -8.0},
                {txtImageMult, 0.5},
                {txtImageOffset, 127.0}
            };
        }

        object[,] DataMap;
                        
        private void OnParamTextLeave(object sender, EventArgs e)
        {
            int i;
            for (i = 0; i < DataMap.GetLength(0); i++)
            {
                if (sender == DataMap[i, 0])
                    try
                    {
                        DataMap[i, 1] = Convert.ToDouble(((TextBox)sender).Text, System.Globalization.CultureInfo.InvariantCulture);
                    }
                    catch (Exception x)
                    {
                        ((TextBox)sender).Text = ((double)DataMap[i, 1]).ToString(System.Globalization.CultureInfo.InvariantCulture);
                    }                
            }
        }

        internal double m_FilterMult;
        internal double m_FilterZeroThresh;
        internal double m_ImageMult;
        internal double m_ImageOffset;

        private void btnOK_Click(object sender, EventArgs e)
        {
            m_FilterMult = (double)DataMap[0, 1];
            m_FilterZeroThresh = (double)DataMap[1, 1];
            m_ImageMult = (double)DataMap[2, 1];
            m_ImageOffset = (double)DataMap[3, 1];
            DialogResult = DialogResult.OK;
            Close();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        private void OnLoad(object sender, EventArgs e)
        {
            int i;
            for (i = 0; i < DataMap.GetLength(0); i++)
            {
                OnParamTextLeave(DataMap[i, 0], null);
            }
        }
    }
}