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
    public partial class MainForm : Form
    {
        SySal.OperaDb.OperaDbConnection NewConn
        {
            get
            {
                SySal.OperaDb.OperaDbConnection conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
                conn.Open();
                return conn;
            }
        }

        public MainForm()
        {
            InitializeComponent();
        }

        private void MainForm_Load(object sender, EventArgs e)
        {
            OperaDbConnection conn = null;
            try
            {
                conn = NewConn;
                if (SySal.OperaDb.Convert.ToInt32(new OperaDbCommand("SELECT count(*) FROM ALL_TABLES WHERE OWNER='OPERA' AND UPPER(TABLE_NAME) = 'XT_BRICK_TASKS'", conn).ExecuteScalar()) == 1 &&
                    SySal.OperaDb.Convert.ToInt32(new OperaDbCommand("SELECT count(*) FROM ALL_TABLES WHERE OWNER='OPERA' AND UPPER(TABLE_NAME) = 'XT_BRICK_FILES'", conn).ExecuteScalar()) == 1) 
                {
                    btnAddLabManagerSupport.Visible = false;
                    btnAddLabManagerSupport.Enabled = false;
                }
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Initialization error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (conn != null) conn.Close();
            }
            btnRefresh_Click(null, null);
        }

        private void btnRefresh_Click(object sender, EventArgs e)
        {
            txtNotes.Text = "";
            txtStatus.Text = "";
            txtType.Text = "";
            txtResultStatus.Text = "";
            box2011.Checked = false;
            box2008.Checked = false;
            box2009.Checked = false;
            box2010.Checked = false;
            boxAll.Checked = true;
            lvOps.Items.Clear();
            lvBricks.Items.Clear();
            lvSecondBrick.Items.Clear();
            txtFilterEvent.Text = "";
            cmbOpType.Items.Clear();
            OperaDbConnection conn = null;
            lvBricks.BeginUpdate();
            try
            {
                conn = NewConn;
                System.Data.DataSet dsb = new DataSet();
                new OperaDbDataAdapter("SELECT ID_EVENTBRICK, OPTYPE FROM OPERA.xt_brick_tasks WHERE (ID_EVENTBRICK, ORDER#) IN (SELECT ID_EVENTBRICK, MAX(ORDER#) FROM OPERA.xt_brick_tasks GROUP BY ID_EVENTBRICK) " + (chkToDo.Checked ? " AND OPTYPE <> 'CLOSE'" : "") + ((txtFilter.Text.Trim().Length > 0) ? (" AND ID_EVENTBRICK " + txtFilter.Text) : "") + " ORDER BY ID_EVENTBRICK", conn).Fill(dsb);
                foreach (System.Data.DataRow dr in dsb.Tables[0].Rows)
                    lvBricks.Items.Add(dr[0].ToString()).SubItems.Add(dr[1].ToString());
                System.Data.DataSet dsot = new DataSet();
                new OperaDbDataAdapter("SELECT DISTINCT OPTYPE FROM OPERA.xt_brick_tasks ORDER BY OPTYPE", conn).Fill(dsot);
                foreach (System.Data.DataRow dr1 in dsot.Tables[0].Rows)
                    cmbOpType.Items.Add(dr1[0].ToString());
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "DB error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                lvBricks.EndUpdate();
                if (conn != null) conn.Close();
            }
           
        }

        private void OnSelBrick(object sender, EventArgs e)
        {
            lvOps.Items.Clear();
            lvSecondBrick.Items.Clear();
            RefreshFiles();
            txtNotes.Text = "";
            if (lvBricks.SelectedItems.Count == 1)
            {
                OperaDbConnection conn = null;
                lvOps.BeginUpdate();
                lvSecondBrick.BeginUpdate();
                try
                {
                    conn = NewConn;
                    System.Data.DataSet ds = new DataSet();
                    new OperaDbDataAdapter("SELECT ORDER#, OPTYPE, ID_PROCESSOPERATION, NOTES FROM OPERA.xt_brick_tasks WHERE ID_EVENTBRICK = " + lvBricks.SelectedItems[0].SubItems[0].Text + " ORDER BY ORDER#", conn).Fill(ds);
                    foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                    {
                        ListViewItem lvi = new ListViewItem(dr[1].ToString());
                        lvi.Tag = dr[0].ToString();
                        lvi.SubItems.Add(dr[2].ToString());
                        lvi.SubItems.Add(dr[3].ToString());
                        lvOps.Items.Add(lvi);
                    }
                    //REGINA MOD START
                    System.Data.DataSet dssec = new DataSet();
                    new OperaDbDataAdapter("select event,  id_eventbrick, Locstatus from OPERA.xt_vertexloc_status where event = (select event from OPERA.xt_vertexloc_status where id_eventbrick = " + lvBricks.SelectedItems[0].SubItems[0].Text +")", conn).Fill(dssec);
                    foreach (System.Data.DataRow dr2 in dssec.Tables[0].Rows)
                    {
                        ListViewItem lvy = new ListViewItem(dr2[0].ToString());
                        lvy.SubItems.Add(dr2[1].ToString());
                        lvy.SubItems.Add(dr2[2].ToString());
                        
                    
                          lvSecondBrick.Items.Add(lvy);
                          
                       
                    }
                    System.Data.DataSet dsstatus = new DataSet();
                    new OperaDbDataAdapter(" select result_status from (select idb from (select distinct(id_eventbrick) as idb from OPERA.xt_brick_tasks) where idb between 1000000 and 1999999)left join OPERA.tv_cs_results on (mod(idb,1000000) = mod(id_cs_eventbrick,1000000)) where idb = " + lvBricks.SelectedItems[0].SubItems[0].Text, conn).Fill(dsstatus);
                    foreach (System.Data.DataRow dt in dsstatus.Tables[0].Rows)
                        txtResultStatus.Text = dt[0].ToString();

                    System.Data.DataSet dstype = new DataSet();
                    new OperaDbDataAdapter("select evtype from xt_vertexloc_status where id_eventbrick = " + lvBricks.SelectedItems[0].SubItems[0].Text, conn).Fill(dstype);
                    foreach (System.Data.DataRow dty in dstype.Tables[0].Rows)
                        txtType.Text = dty[0].ToString();

                }
                
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "DB error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    lvOps.EndUpdate();
                    lvSecondBrick.EndUpdate();
                    if (conn != null) conn.Close();
                }
            }
        }

        private void OnSelOp(object sender, EventArgs e)
        {
            txtNotes.Text = "";
            if (lvOps.SelectedItems.Count == 1)
            {
                txtNotes.Text = lvOps.SelectedItems[0].SubItems[2].Text;
                RefreshFiles();
            }
        }

        private void btnUpdateNotes_Click(object sender, EventArgs e)
        {
            if (lvOps.SelectedItems.Count == 1)
            {
                OperaDbConnection conn = null;
                try
                {
                    conn = NewConn;
                    OperaDbCommand cmd = new OperaDbCommand("UPDATE OPERA.xt_brick_tasks SET NOTES = :nt WHERE ID_EVENTBRICK = " + lvBricks.SelectedItems[0].SubItems[0].Text + " AND ORDER# = " + lvOps.SelectedItems[0].Tag.ToString(), conn);
                    cmd.Parameters.Add("nt", OperaDbType.CLOB, ParameterDirection.Input).Value = txtNotes.Text;
                    cmd.ExecuteNonQuery();
                    lvOps.SelectedItems[0].SubItems[2].Text = txtNotes.Text;
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "DB error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    txtNotes.Text = lvOps.SelectedItems[0].SubItems[2].Text;
                }
                finally
                {
                    if (conn != null) conn.Close();
                }
            }
        }

        private void btnCloseBrick_Click(object sender, EventArgs e)
        {
            if (lvBricks.SelectedItems.Count == 1)
            {
                OperaDbConnection conn = null;
                try
                {
                    conn = NewConn;
                    new OperaDbCommand("INSERT INTO OPERA.xt_brick_tasks (ID_EVENTBRICK, ORDER#, OPTYPE) (SELECT ID_EVENTBRICK, NEWORD#, 'CLOSE' FROM (SELECT ID_EVENTBRICK, MAX(ORDER#) + 1 as NEWORD# FROM OPERA.xt_brick_tasks WHERE ID_EVENTBRICK = " + lvBricks.SelectedItems[0].SubItems[0].Text + " GROUP BY ID_EVENTBRICK))", conn).ExecuteNonQuery();
                    conn.Close(); conn = null;
                    btnRefresh_Click(null, null);
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "DB error", MessageBoxButtons.OK, MessageBoxIcon.Error);                    
                }
                finally
                {
                    if (conn != null) conn.Close();
                }
            }
        }

        private void btnReceiveBrick_Click(object sender, EventArgs e)
        {
            long idbrick = 0;
            try
            {
                idbrick = System.Convert.ToInt64(txtRecvBrickId.Text);
                if (idbrick < 1000000) throw new Exception("OPERA bricks must have a code starting with 1000000.");
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Input error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            OperaDbConnection conn = null;
            try
            {
                conn = NewConn;
                new OperaDbCommand("INSERT INTO OPERA.xt_brick_tasks (ID_EVENTBRICK, ORDER#, OPTYPE) VALUES (" + idbrick + ",1,'RECEIVE')", conn).ExecuteNonQuery();
                conn.Close(); 
                conn = null;
                btnRefresh_Click(null, null);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "DB error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (conn != null) conn.Close();
            }
        }

        private void btnImportAutoOp_Click(object sender, EventArgs e)
        {
            OperaDbConnection conn = null;
            try
            {
                conn = NewConn;
                ProcOpForm pof = new ProcOpForm(conn, lvBricks.SelectedItems[0].Text);
                OperaDbCommand cmd = new OperaDbCommand("INSERT INTO OPERA.xt_brick_tasks (ID_EVENTBRICK, ORDER#, OPTYPE, ID_PROCESSOPERATION, NOTES) (SELECT ID_EVENTBRICK, NEWORD#, :opty, :procop, :notes FROM (SELECT ID_EVENTBRICK, MAX(ORDER#) + 1 as NEWORD# FROM OPERA.xt_brick_tasks WHERE ID_EVENTBRICK = " + lvBricks.SelectedItems[0].Text + " GROUP BY ID_EVENTBRICK))", conn);
                cmd.Parameters.Add("opty", OperaDbType.String, ParameterDirection.Input);
                cmd.Parameters.Add("procop", OperaDbType.String, ParameterDirection.Input);
                cmd.Parameters.Add("notes", OperaDbType.CLOB, ParameterDirection.Input);
                if (pof.ShowDialog() == DialogResult.OK)
                {
                    foreach (string[] ss in pof.m_OutIds)
                    {
                        cmd.Parameters[0].Value = ss[1];
                        cmd.Parameters[1].Value = ss[0];
                        cmd.Parameters[2].Value = ss[2];
                        cmd.ExecuteNonQuery();
                    }
                }                
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "DB error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (conn != null) conn.Close();
                btnRefresh_Click(null, null);
            }            
        }

        private void btnAddManualOp_Click(object sender, EventArgs e)
        {
            if (lvBricks.SelectedItems.Count == 1 && cmbOpType.Text.Trim().Length > 0)
            {
                OperaDbConnection conn = null;
                try
                {
                    conn = NewConn;
                    OperaDbCommand cmd = new OperaDbCommand("INSERT INTO OPERA.xt_brick_tasks (ID_EVENTBRICK, ORDER#, OPTYPE, NOTES) (SELECT ID_EVENTBRICK, NEWORD#, :opty, :notes FROM (SELECT ID_EVENTBRICK, MAX(ORDER#) + 1 as NEWORD# FROM OPERA.xt_brick_tasks WHERE ID_EVENTBRICK = " + lvBricks.SelectedItems[0].Text + " GROUP BY ID_EVENTBRICK))", conn);
                    cmd.Parameters.Add("opty", OperaDbType.String, ParameterDirection.Input).Value = cmbOpType.Text.Trim();
                    cmd.Parameters.Add("notes", OperaDbType.CLOB, ParameterDirection.Input).Value = txtNotes.Text;
                    cmd.ExecuteNonQuery();
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "DB error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    if (conn != null) conn.Close();
                    btnRefresh_Click(null, null);
                }
            }
        }

        private void btnPubBrick_Click(object sender, EventArgs e)
        {
            if (lvBricks.SelectedItems.Count == 1)
            {
                OperaDbConnection conn = null;
                try
                {
                    conn = NewConn;
                    PubForm pf = new PubForm(conn, lvBricks.SelectedItems[0].SubItems[0].Text);
                    pf.ShowDialog();
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    if (conn != null) conn.Close();
                    btnRefresh_Click(null, null);
                }
            }
        }

        private void btnAddLabManagerSupport_Click(object sender, EventArgs e)
        {
            OperaDbConnection conn = null;
            try
            {
                conn = NewConn;
                foreach (string s in CmdSQL)
                    try
                    {
                        new OperaDbCommand(s, conn).ExecuteNonQuery();
                    }
                    catch (Exception x)
                    {
                        if (MessageBox.Show(x.ToString(), "Continue?", MessageBoxButtons.YesNo, MessageBoxIcon.Warning) == DialogResult.No) throw x;
                    }
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "DB initialization error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (conn != null) conn.Close();                
            }
        }

        static string[] CmdSQL = new string[] {
            "CREATE TABLE OPERA.XT_BRICK_TASKS " +
            " ( ID_EVENTBRICK NUMBER(*,0) NOT NULL, " +
            " ORDER# NUMBER(*,0) NOT NULL, " + 
            " OPTYPE VARCHAR2(64 BYTE), " +
            " ID_PROCESSOPERATION NUMBER(*,0), " +
            "NOTES CLOB, " +
            "CONSTRAINT XK_BRICK_TASKS PRIMARY KEY (ID_EVENTBRICK, ORDER#) USING INDEX TABLESPACE OPERASYSTEM " +
            ") TABLESPACE OPERASYSTEM",
            "CREATE INDEX OPERA.XX_BRICK_TASKS_OPTYPES ON OPERA.XT_BRICK_TASKS (OPTYPE, ID_EVENTBRICK) TABLESPACE OPERASYSTEM",
            "grant select on XT_BRICK_TASKS to RL_DATAREADER",
            "grant insert on XT_BRICK_TASKS to RL_DATAWRITER",
            "grant update on XT_BRICK_TASKS to RL_DATAWRITER",
            "grant delete on XT_BRICK_TASKS to RL_DATAWRITER",
            "CREATE TABLE OPERA.XT_BRICK_FILES (" +
            "ID_EVENTBRICK NUMBER(*,0) NOT NULL, " +
	        "ORDER# NUMBER(*,0) NOT NULL, " + 
	        "FILENAME VARCHAR2(256 BYTE), " +
	        "FILEDATA BLOB, " +
	        "CONSTRAINT XK_BRICK_FILES PRIMARY KEY (ID_EVENTBRICK, ORDER#, FILENAME) USING INDEX TABLESPACE OPERASYSTEM, " +
            "CONSTRAINT XF_BRICK_FILES FOREIGN KEY (ID_EVENTBRICK, ORDER#) REFERENCES OPERA.XT_BRICK_TASKS (ID_EVENTBRICK, ORDER#) " +
            ") TABLESPACE OPERASYSTEM",
            "CREATE INDEX OPERA.XF_BRICK_FILES ON OPERA.XT_BRICK_FILES (ID_EVENTBRICK, ORDER#) TABLESPACE OPERASYSTEM",
            "grant select on XT_BRICK_FILES to RL_DATAREADER",
            "grant insert on XT_BRICK_FILES to RL_DATAWRITER",
            "grant update on XT_BRICK_FILES to RL_DATAWRITER",
            "grant delete on XT_BRICK_FILES to RL_DATAWRITER",
        };

        void RefreshFiles()
        {
            lvFiles.Items.Clear();
            int i;
            for (i = 2; i < imgFiles.Images.Count; )
                imgFiles.Images.RemoveAt(2);
            if (lvOps.SelectedItems.Count != 1) return;
            OperaDbConnection conn = null;
            try
            {
                conn = NewConn;
                System.Data.DataSet ds = new DataSet();
                new SySal.OperaDb.OperaDbDataAdapter("SELECT FILENAME, FILEDATA FROM OPERA.xt_brick_files WHERE ID_EVENTBRICK = " + lvBricks.SelectedItems[0].SubItems[0].Text + " AND ORDER# = " + lvOps.SelectedItems[0].Tag.ToString(), conn).Fill(ds);
                foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                {
                    ListViewItem lvi = lvFiles.Items.Add(dr[0].ToString());
                    lvi.Tag = (byte[])dr[1];
                    string fname = dr[0].ToString().ToLower();
                    if (fname.EndsWith(".txt"))
                    {
                        lvi.ImageIndex = 0;
                    }
                    else if (fname.EndsWith(".x3l"))
                    {
                        lvi.ImageIndex = 1;
                    }
                    else if (fname.EndsWith(".gif") || fname.EndsWith(".jpg") || fname.EndsWith(".jpeg") || fname.EndsWith(".png"))
                    {
                        System.IO.MemoryStream mstr = new System.IO.MemoryStream((byte[])lvi.Tag);
                        System.Drawing.Image img = System.Drawing.Image.FromStream(mstr);
                        lvi.ImageIndex = imgFiles.Images.Add(img, Color.White);
                    }
                }
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "DB initialization error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (conn != null) conn.Close();
            }
        }

        private void btnImportFile_Click(object sender, EventArgs e)
        {
            if (lvOps.SelectedItems.Count != 1) return;
            if (ofnSelectImportFile.ShowDialog() == DialogResult.OK)
            {
                string fname = ofnSelectImportFile.FileName;
                byte[] data = null;
                System.IO.FileStream r = null;
                try
                {
                    r = new System.IO.FileStream(fname, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.Read);
                    data = new byte[r.Length];
                    r.Read(data, 0, data.Length);
                    r.Close();
                    r = null;
                }
                catch (Exception x)
                {
                    if (r != null) r.Close();
                    data = null;
                    MessageBox.Show(x.ToString(), "Can't load file.", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
                OperaDbConnection conn = null;
                try
                {
                    conn = NewConn;
                    SySal.OperaDb.OperaDbCommand cmd = new OperaDbCommand("INSERT INTO OPERA.xt_brick_files (ID_EVENTBRICK, ORDER#, FILENAME, FILEDATA) VALUES (" + lvBricks.SelectedItems[0].SubItems[0].Text + "," + lvOps.SelectedItems[0].Tag.ToString() + ",:a,:b)", conn);
                    cmd.Parameters.Add("a", OperaDbType.String, ParameterDirection.Input).Value = fname.Substring(fname.LastIndexOfAny(new char[] { '\\', '/' }) + 1);;
                    cmd.Parameters.Add("b", OperaDbType.BLOB, ParameterDirection.Input).Value = data;
                    cmd.ExecuteNonQuery();                    
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "Insertion error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    if (conn != null) conn.Close();
                }
                RefreshFiles();
            }
        }

        private void btnRemoveFile_Click(object sender, EventArgs e)
        {
            if (lvFiles.SelectedItems.Count != 1) return;
            if (MessageBox.Show("Are you sure you want to remove the selected file?", "Confirmation needed", MessageBoxButtons.YesNo, MessageBoxIcon.Warning) == DialogResult.Yes)
            {
                OperaDbConnection conn = null;
                try
                {
                    conn = NewConn;
                    SySal.OperaDb.OperaDbCommand cmd = new OperaDbCommand("DELETE FROM OPERA.xt_brick_files WHERE ID_EVENTBRICK = " + lvBricks.SelectedItems[0].SubItems[0].Text + " AND ORDER# = " + lvOps.SelectedItems[0].Tag.ToString() + " AND FILENAME = :a", conn);
                    cmd.Parameters.Add("a", OperaDbType.String, ParameterDirection.Input).Value = lvFiles.SelectedItems[0].SubItems[0].Text;
                    cmd.ExecuteNonQuery();                    
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "Deletion error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    if (conn != null) conn.Close();
                }
                RefreshFiles();                
            }
        }

        private void OnFileSel(object sender, EventArgs e)
        {
            if (lvFiles.SelectedItems.Count != 1)
            {
                pcBox.Image = null;
                pcBox.Visible = false;
                rtbText.Visible = false;
                btnInfinity.Visible = btnZoomIn.Visible = btnZoomOut.Visible = gdiDisp.Visible = false;                
            }
            else
            {
                if (lvFiles.SelectedItems[0].ImageIndex == 0)
                {
                    rtbText.Text = new System.IO.StreamReader(new System.IO.MemoryStream((byte[])lvFiles.SelectedItems[0].Tag)).ReadToEnd();
                    pcBox.Image = null;
                    pcBox.Visible = false;
                    rtbText.Visible = true;
                    btnInfinity.Visible = btnZoomIn.Visible = btnZoomOut.Visible = gdiDisp.Visible = false;
                }
                if (lvFiles.SelectedItems[0].ImageIndex == 1)
                {
                    string fname = System.Environment.ExpandEnvironmentVariables("%TEMP%\\" + System.Guid.NewGuid().ToString() + ".x3l");
                    System.IO.FileStream w = null;
                    try
                    {
                        w = new System.IO.FileStream(fname, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite, System.IO.FileShare.None);
                        byte [] b = (byte[])lvFiles.SelectedItems[0].Tag;
                        w.Write(b, 0, b.Length);
                        w.Flush();
                        w.Close();
                        w = null;
                        gdiDisp.LoadScene(fname);
                        System.IO.File.Delete(fname);
                    }
                    catch (Exception x)
                    {
                        if (w != null) w.Close();
                        w = null;
                        try
                        {
                            System.IO.File.Delete(fname);
                        }
                        catch(Exception) {}
                        MessageBox.Show(x.ToString(), "Can't create temporary file.", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                    pcBox.Image = null;
                    pcBox.Visible = false;
                    rtbText.Visible = false;
                    btnInfinity.Visible = btnZoomIn.Visible = btnZoomOut.Visible = gdiDisp.Visible = true;
                }
                else if (lvFiles.SelectedItems[0].ImageIndex >= 2)
                {
                    pcBox.Image = System.Drawing.Image.FromStream(new System.IO.MemoryStream((byte[])lvFiles.SelectedItems[0].Tag));
                    pcBox.Visible = true;
                    rtbText.Visible = false;
                    btnInfinity.Visible = btnZoomIn.Visible = btnZoomOut.Visible = gdiDisp.Visible = false;
                }
            }
        }

        private void btnSaveFile_Click(object sender, EventArgs e)
        {
            if (lvFiles.SelectedItems.Count != 1) return;
            sfSelectSaveFile.FileName = lvFiles.SelectedItems[0].SubItems[0].Text;
            if (sfSelectSaveFile.ShowDialog() == DialogResult.OK)
            {
                System.IO.FileStream w = null;
                try
                {
                    w = new System.IO.FileStream(sfSelectSaveFile.FileName, System.IO.FileMode.Create, System.IO.FileAccess.Write);
                    byte[] b = (byte[])lvFiles.SelectedItems[0].Tag;
                    w.Write(b, 0, b.Length);
                    w.Flush();
                    w.Close();
                    w = null;
                }
                catch (Exception x)
                {
                    if (w != null) w.Close();
                    w = null;
                    MessageBox.Show(x.ToString(), "Can't save file.", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        private void btnZoomIn_Click(object sender, EventArgs e)
        {
            gdiDisp.Zoom = 1.1 * gdiDisp.Zoom;
            gdiDisp.Render();
        }

        private void btnZoomOut_Click(object sender, EventArgs e)
        {
            gdiDisp.Zoom = gdiDisp.Zoom / 1.1;
            gdiDisp.Render();
        }

        private void btnInfinity_Click(object sender, EventArgs e)
        {
            gdiDisp.Infinity = !gdiDisp.Infinity;
            gdiDisp.Render();
        }

        private void btnDeleteOp_Click(object sender, EventArgs e)
        {
            if (lvOps.SelectedItems.Count == 1 && MessageBox.Show("Deletion cannot be undone.\r\nAll notes and related files will be lost.\r\nDB data will be untouched.\r\nConfirm deletion?", "Confirmation required", MessageBoxButtons.YesNo, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2) == DialogResult.Yes)
            {
                OperaDbConnection conn = null;
                try
                {
                    conn = NewConn;
                    OperaDbTransaction trans = conn.BeginTransaction();
                    new OperaDbCommand("DELETE FROM OPERA.xt_brick_files WHERE ID_EVENTBRICK = " + lvBricks.SelectedItems[0].SubItems[0].Text + " AND ORDER# = " + lvOps.SelectedItems[0].Tag.ToString(), conn).ExecuteNonQuery();
                    new OperaDbCommand("DELETE FROM OPERA.xt_brick_tasks WHERE ID_EVENTBRICK = " + lvBricks.SelectedItems[0].SubItems[0].Text + " AND ORDER# = " + lvOps.SelectedItems[0].Tag.ToString(), conn).ExecuteNonQuery();
                    trans.Commit();
                    btnRefresh_Click(null, null);
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "DB error", MessageBoxButtons.OK, MessageBoxIcon.Error);                    
                }
                finally
                {
                    if (conn != null) conn.Close();
                }
            }

        }

        private void btnSearchEvent_Click(object sender, EventArgs e)
        {
            
            lvOps.Items.Clear();
            lvBricks.Items.Clear();
            lvSecondBrick.Items.Clear();
            cmbOpType.Items.Clear();
            OperaDbConnection conn = null;
            lvBricks.BeginUpdate();
            try
            {
                conn = NewConn;
                System.Data.DataSet dsb = new DataSet();
                new OperaDbDataAdapter("SELECT ID_EVENTBRICK, OPTYPE FROM OPERA.xt_brick_tasks WHERE (ID_EVENTBRICK, ORDER#) IN (SELECT ID_EVENTBRICK, MAX(ORDER#) FROM OPERA.xt_brick_tasks GROUP BY ID_EVENTBRICK) AND id_eventbrick in (select id_eventbrick from OPERA.xt_vertexloc_status where  event = " + txtFilterEvent.Text + ") ORDER BY id_eventbrick", conn).Fill(dsb);
                foreach (System.Data.DataRow dr in dsb.Tables[0].Rows)
                    lvBricks.Items.Add(dr[0].ToString()).SubItems.Add(dr[1].ToString());
                
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "DB error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                lvBricks.EndUpdate();
                if (conn != null) conn.Close();
            }
           
        

        }

        private void btnSearchStatus_Click(object sender, EventArgs e)
        {

            lvOps.Items.Clear();
            lvBricks.Items.Clear();
            lvSecondBrick.Items.Clear();
            cmbOpType.Items.Clear();
            OperaDbConnection conn = null;
            lvBricks.BeginUpdate();
            try
            {
                conn = NewConn;
                System.Data.DataSet dsb = new DataSet();
                if (boxAll.Checked)
                    new OperaDbDataAdapter("SELECT ID_EVENTBRICK, OPTYPE FROM OPERA.xt_brick_tasks WHERE (ID_EVENTBRICK, ORDER#) IN (SELECT ID_EVENTBRICK, MAX(ORDER#) FROM OPERA.xt_brick_tasks GROUP BY ID_EVENTBRICK) AND OPTYPE like '" + txtStatus.Text + "' ORDER BY id_eventbrick", conn).Fill(dsb);
                else if (box2008.Checked)
                    new OperaDbDataAdapter("SELECT ID_EVENTBRICK, OPTYPE FROM OPERA.xt_brick_tasks WHERE (ID_EVENTBRICK, ORDER#) IN (SELECT ID_EVENTBRICK, MAX(ORDER#) FROM OPERA.xt_brick_tasks GROUP BY ID_EVENTBRICK) AND OPTYPE like '" + txtStatus.Text + "' AND id_eventbrick in (select id_eventbrick from OPERA.xt_vertexloc_status where  runyear =2008) ORDER BY id_eventbrick", conn).Fill(dsb);
                else if (box2009.Checked)
                    new OperaDbDataAdapter("SELECT ID_EVENTBRICK, OPTYPE FROM OPERA.xt_brick_tasks WHERE (ID_EVENTBRICK, ORDER#) IN (SELECT ID_EVENTBRICK, MAX(ORDER#) FROM OPERA.xt_brick_tasks GROUP BY ID_EVENTBRICK) AND OPTYPE like '" + txtStatus.Text + "' AND id_eventbrick in (select id_eventbrick from OPERA.xt_vertexloc_status where  runyear =2009) ORDER BY id_eventbrick", conn).Fill(dsb);
                else if (box2010.Checked)
                    new OperaDbDataAdapter("SELECT ID_EVENTBRICK, OPTYPE FROM OPERA.xt_brick_tasks WHERE (ID_EVENTBRICK, ORDER#) IN (SELECT ID_EVENTBRICK, MAX(ORDER#) FROM OPERA.xt_brick_tasks GROUP BY ID_EVENTBRICK) AND OPTYPE like '" + txtStatus.Text + "' AND id_eventbrick in (select id_eventbrick from OPERA.xt_vertexloc_status where  runyear =2010) ORDER BY id_eventbrick", conn).Fill(dsb);
                else if (box2011.Checked)
                    new OperaDbDataAdapter("SELECT ID_EVENTBRICK, OPTYPE FROM OPERA.xt_brick_tasks WHERE (ID_EVENTBRICK, ORDER#) IN (SELECT ID_EVENTBRICK, MAX(ORDER#) FROM OPERA.xt_brick_tasks GROUP BY ID_EVENTBRICK) AND OPTYPE like '" + txtStatus.Text + "' AND id_eventbrick in (select id_eventbrick from OPERA.xt_vertexloc_status where  runyear =2011) ORDER BY id_eventbrick", conn).Fill(dsb);
                                 
                
                foreach (System.Data.DataRow dr in dsb.Tables[0].Rows)
                    lvBricks.Items.Add(dr[0].ToString()).SubItems.Add(dr[1].ToString());


            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "DB error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                lvBricks.EndUpdate();
                if (conn != null) conn.Close();
            }

        }

        private void box2008_CheckedChanged(object sender, EventArgs e)
        {
            boxAll.Checked = false;
            box2010.Checked = false;
            box2011.Checked = false;
            box2009.Checked = false;
           

               
        }

        private void box2009_CheckedChanged(object sender, EventArgs e)
        {
            boxAll.Checked = false;
            box2008.Checked = false;
            box2010.Checked = false;
            box2011.Checked = false;
            

           

        }

        private void box2010_CheckedChanged(object sender, EventArgs e)
        {
            boxAll.Checked = false;
            box2008.Checked = false;
            box2009.Checked = false;
            box2011.Checked = false;

        }

        private void box2011_CheckedChanged(object sender, EventArgs e)
        {
            boxAll.Checked = false;
            box2008.Checked = false;
            box2009.Checked = false;
            box2010.Checked = false;
        

        }

      
      

       
    
        

        
    }

}
