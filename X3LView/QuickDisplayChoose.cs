using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.X3LView
{
    public partial class QuickDisplayChoose : Form
    {
        public string m_Scene = null;

        public QuickDisplayChoose()
        {
            InitializeComponent();
        }

        private void OnLoad(object sender, EventArgs e)
        {
            SySal.OperaDb.OperaDbConnection Conn = null;
            try
            {
                Conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
                Conn.Open();
                System.Data.DataSet dssc = new DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT ID FROM TB_EVENTBRICKS WHERE ID BETWEEN 1000000 AND 9999999 ORDER BY ID", Conn).Fill(dssc);
                foreach (System.Data.DataRow dr in dssc.Tables[0].Rows)
                    comboCS.Items.Add(dr[0].ToString());
            }
            catch (Exception) { Close(); }
            finally
            {
                if (Conn != null) Conn.Close();
            }
        }

        private void OKScanDisplayButton_Click(object sender, EventArgs e)
        {
            if (comboCS.SelectedItem == null) return;
            SySal.OperaDb.OperaDbConnection Conn = null;
            try
            {
                Conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
                Conn.Open();
                new SySal.OperaDb.OperaDbCommand("ALTER SESSION SET NLS_NUMERIC_CHARACTERS = '.,'", Conn).ExecuteNonQuery();
                m_Scene = new SySal.OperaDb.OperaDbCommand("SELECT FN_SCAN_DISPLAY(" + comboCS.SelectedItem.ToString() + ") FROM DUAL", Conn).ExecuteScalar().ToString();
                DialogResult = DialogResult.OK;
                Close();
            }
            catch (Exception x) 
            {
                MessageBox.Show(x.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);            
            }
            finally
            {
                if (Conn != null) Conn.Close();
            }
        }
    }
}