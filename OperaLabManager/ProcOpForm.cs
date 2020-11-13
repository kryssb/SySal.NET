using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using SySal;
using SySal.OperaDb;

namespace SySal.Executables.OperaLabManager
{
    public partial class ProcOpForm : Form
    {
        string m_BrickId;

        OperaDbConnection Conn;

        public ProcOpForm(OperaDbConnection conn, string brickid)
        {
            InitializeComponent();
            Conn = conn;
            m_BrickId = brickid;
        }

        private void OnLoad(object sender, EventArgs e)
        {            
            try
            {
                System.Data.DataSet ds = new DataSet();
                new OperaDbDataAdapter("select procopid, case when nvol > 0 then 'TS' when nsb > 0 then decode(maxord, 0, 'SF', 'SB') when nfbk > 0 then 'FB' else 'UNKNOWN' end as optype, notes from " +
                    "(SELECT PROCOPID, NOTES, sumplates, maxord, NVOL, NSB, sum(nvl2(id_reconstruction,1,0)) as NFBK FROM (SELECT idb, PROCOPID, NOTES, maxord, sumplates, NVOL, sum(nvl2(path,1,0)) as NSB FROM (SELECT idb, PROCOPID, NOTES, maxord, sumplates, sum(nvl2(volume,1,0)) as NVOL FROM ( " +
                    "select idb, procopid, notes1 as notes, max(idord - plateord) as maxord,sum(nvl(id_plate,0)) as sumplates from (select idb, procopid, notes1, row_number() over (partition by procopid order by id) as idord, row_number() over (partition by procopid order by id_plate, id) as plateord, id_plate from(select id_eventbrick as idb, id as procopid, notes as notes1 from tb_proc_operations where id_eventbrick = " + m_BrickId + " and success = 'Y' and id_parent_operation is null) left join tb_proc_operations on (id_parent_operation = procopid)) group by idb, procopid, notes1 " +
                    ") LEFT JOIN TB_VOLUMES ON (ID_EVENTBRICK = idb AND ID_PROCESSOPERATION = PROCOPID) GROUP BY idb, PROCOPID, NOTES, maxord, sumplates) LEFT JOIN TB_SCANBACK_PATHS ON (ID_EVENTBRICK = idb AND ID_PROCESSOPERATION = PROCOPID) GROUP BY idb, PROCOPID, NOTES, NVOL, maxord, sumplates) LEFT JOIN VW_FEEDBACK_RECONSTRUCTIONS ON (ID_EVENTBRICK = idb AND ID_PROCESSOPERATION = PROCOPID) GROUP BY PROCOPID, NOTES, NVOL, NSB, maxord, sumplates) " +
                    "order by procopid", Conn).Fill(ds);
                foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                {
                    ListViewItem lvi = lvProcOps.Items.Add(dr[0].ToString());
                    lvi.SubItems.Add(dr[1].ToString());
                    lvi.SubItems.Add(dr[2].ToString());                    
                }
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "DB error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        private void btnOk_Click(object sender, EventArgs e)
        {
            if (lvProcOps.SelectedItems.Count < 1) return;
            m_OutIds = new string[lvProcOps.SelectedItems.Count][];
            int i;
            for (i = 0; i < lvProcOps.SelectedItems.Count; i++)
            {
                m_OutIds[i] = new string[3];
                m_OutIds[i][0] = lvProcOps.SelectedItems[i].SubItems[0].Text;
                m_OutIds[i][1] = lvProcOps.SelectedItems[i].SubItems[1].Text;
                m_OutIds[i][2] = lvProcOps.SelectedItems[i].SubItems[2].Text;
            }
            DialogResult = DialogResult.OK;
        }

        public string[][] m_OutIds;
    }
}