using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.EasyReconstruct
{
    public partial class QBrowser : Form
    {
        public QBrowser(string title, string info)
        {
            InitializeComponent();

            txtInfo.Text = info;
            this.Text = title;
        }
    }
}