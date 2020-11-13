using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using SySal.OperaDb;
using System.IO;

namespace OperaBrickManager
{
    public partial class MainForm : Form
    {
        OperaDbConnection DBConn = OperaDbCredentials.CreateFromRecord().Connect();        

        public MainForm()
        {
            InitializeComponent();
        }

        private void OnLoad(object sender, EventArgs e)
        {
            try
            {
                DBConn.Open();
                SySal.OperaDb.Schema.DB = DBConn;
            }
            catch (Exception x)
            {
                DBConn.Close();
                MessageBox.Show(x.Message, "Error opening DB connection!", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Close();
            }
            try
            {
                RefreshBricksSets();
                if (lvBrickSets.Items.Count != 0)
                {
                    lvBrickSets.Items[0].Selected = true;
                    GetBricksInBrickSet(lvBrickSets.SelectedItems[0].Tag.ToString());
                }
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Error opening DB connection!", MessageBoxButtons.OK, MessageBoxIcon.Error);
                Close();
            }
            
        }

        private void btnRefreshBrickSets_Click(object sender, EventArgs e)
        {
            RefreshBricksSets();
        }

        private void btnAddBrickSet_Click(object sender, EventArgs e)
        {
            AddBrickSetForm ab = new AddBrickSetForm();
            if (ab.ShowDialog() == DialogResult.OK)
            {
                Cursor oldc = Cursor;
                try
                {
                    try
                    {
                        Cursor = Cursors.WaitCursor;
                        SySal.OperaDb.Schema.PC_ADD_BRICK_SET.Call(ab.SetName, ab.RangeMin, ab.RangeMax, ab.TablespaceExt.ToUpper(), ab.DefaultId);
                        btnRefreshBrickSets_Click(sender, e);
                    }
                    finally
                    {
                        Cursor = oldc;
                    }
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "Error inserting brick set!", MessageBoxButtons.OK);
                }
            }
            ab.Dispose();
        }

        private void txtRemoveBrickSet_Click(object sender, EventArgs e)
        {
            if (lvBrickSets.SelectedItems.Count != 1) return;
            if (MessageBox.Show("Are you sure you want to delete the selected brick set?", "Operation warning", MessageBoxButtons.YesNo, MessageBoxIcon.Warning) == DialogResult.Yes)
            {
                Cursor oldc = Cursor;
                try
                {
                    try
                    {
                        Cursor = Cursors.WaitCursor;
                        SySal.OperaDb.Schema.PC_REMOVE_BRICK_SET.Call(lvBrickSets.SelectedItems[0].Tag.ToString());
                        btnRefreshBrickSets_Click(sender, e);
                    }
                    finally
                    {
                        Cursor = oldc;
                    }
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "Error removing brick set!", MessageBoxButtons.OK);
                }                
            }
        }

        private void RefreshBricksSets()
        {
            System.Data.DataSet ds = new DataSet();
            new OperaDbDataAdapter(
                "with sysseg as (select tablespace_name, bytes from sys.dba_segments) " +
                "select setid, id_partition, nbricks, ceil(scansize / 1073741824) as scansizegb, ceil(recsize / 1073741824) as recsizegb, ceil(procsize / 1073741824) as procsizegb from " +
                "(select setid, scansize, recsize, sum(c.bytes) as procsize, id_partition, nbricks from " +
                "(select setid, scansize, sum(b.bytes) as recsize, oprocp, id_partition, nbricks from " +
                "(select setid, sum(a.bytes) as scansize, orecp, oprocp, id_partition, nbricks from " +
                "(select setid, 'OPERASCAN_' || id_partition as oscanp, 'OPERAREC_' || id_partition as orecp, 'OPERAPROC_' || id_partition as oprocp, id_partition, nbricks from " +
                "(select setid, id_partition, sum(nvl2(tb_eventbricks.id, 1, 0)) as nbricks from " +
                "(select id as setid, id_partition from tb_brick_sets) left join tb_eventbricks on (setid = id_set) group by setid, id_partition)) " +
                "inner join sysseg a on (a.tablespace_name = oscanp) group by setid, orecp, oprocp, id_partition, nbricks) " +
                "inner join sysseg b on (b.tablespace_name = orecp) group by setid, scansize, oprocp, id_partition, nbricks) " +
                "inner join sysseg c on (c.tablespace_name = oprocp) group by setid, scansize, recsize, id_partition, nbricks) order by setid"
                , DBConn).Fill(ds);
            lvBrickSets.Items.Clear();
            foreach (DataRow dr in ds.Tables[0].Rows)
            {
                ListViewItem lvi = lvBrickSets.Items.Add(dr[0].ToString());
                lvi.SubItems.Add(dr[1].ToString());
                lvi.SubItems.Add(dr[2].ToString());
                lvi.SubItems.Add(dr[3].ToString());
                lvi.SubItems.Add(dr[4].ToString());
                lvi.SubItems.Add(dr[5].ToString());
                lvi.Tag = dr[0].ToString();
            }
        }

        private void lvBrickSets_DoubleClick(object sender, EventArgs e)
        {
            if (lvBrickSets.Items.Count == 0) return;
            GetBricksInBrickSet(lvBrickSets.SelectedItems[0].Text);
        }

        class BrickTag
        {
            public string Set;
            public int IdInSet;

            public BrickTag(string s, int iis)
            {
                Set = s;
                IdInSet = iis;
            }
        }

        private void GetBricksInBrickSet(String BrickSetId)
        {
            System.Data.DataSet ds = new DataSet();
            new OperaDbDataAdapter("Select id_Set, Id_brick, Minx, MaxX, MinY ,MaxY, MinZ, MaxZ, ZeroX, ZeroY, ZeroZ, id  from tb_eventbricks where id_set =  " + "'" + BrickSetId + "'", 
                DBConn).Fill(ds);
            lvBricks.Items.Clear();
            //int i = 1;
            foreach (DataRow dr in ds.Tables[0].Rows)
            {
                ListViewItem lvi = lvBricks.Items.Add(dr[11].ToString());
                lvi.SubItems.Add(dr[0].ToString());
                lvi.SubItems.Add(dr[1].ToString());
                lvi.SubItems.Add(dr[2].ToString());
                lvi.SubItems.Add(dr[3].ToString());
                lvi.SubItems.Add(dr[4].ToString());
                lvi.SubItems.Add(dr[5].ToString());
                lvi.SubItems.Add(dr[6].ToString());
                lvi.SubItems.Add(dr[7].ToString());
                lvi.SubItems.Add(dr[8].ToString());
                lvi.SubItems.Add(dr[9].ToString());
                lvi.SubItems.Add(dr[10].ToString());
                lvi.Tag = new BrickTag(dr[0].ToString(), System.Convert.ToInt32(dr[1]));                
            }
        }

        private void btnAddBrick_Click(object sender, EventArgs e)
        {
            frmAddBrick ab = new frmAddBrick();

            ab.SetBrick = lvBrickSets.SelectedItems[0].Tag.ToString();

#if true
            if (ab.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    object o = null;
                    SySal.OperaDb.Schema.PC_ADD_BRICK_EASY_Z.Call(ab.Brick, ab.GLMinX, ab.GlMaxX, ab.GlMinY, ab.GlMaxY, ab.NPlate, ab.ZMax, ab.DPlate, ab.SetBrick, ref o, ab.ZeroX, ab.ZeroY);                    
                    ListViewItem lvi = lvBrickSets.SelectedItems[0];
                    lvi.Selected = false;
                    lvi.Selected = true;
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "Error inserting brick!", MessageBoxButtons.OK);
                }
            }

#else           
            if (ab.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    using (StreamWriter file_out = new StreamWriter("C:\\OperaBrickManager\\input1.txt"))
                    {
                        file_out.WriteLine(ab.Brick + "\t" + ab.SetBrick + "\t" + ab.ZMax + "\t" + ab.NPlate + "\t" + ab.DPlate + "\t" + ab.GlMaxX + "\t" + ab.GlMaxY + "\t" + ab.GLMinX + "\t" + ab.GlMinY);
                        file_out.Flush();
                        file_out.Close();
                       
                    }
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "No Write", MessageBoxButtons.OK);
 
                }
                            
            }
#endif                
             ab.Dispose();
            

        }

        private void btnRemoveBrick_Click(object sender, EventArgs e)
        {
            if (lvBricks.SelectedItems.Count != 1)
            {
                MessageBox.Show("You must select one row.", "Selection error");
                return;
            }
#if true
            if (MessageBox.Show("Are you sure you want to delete the selected brick set?", "Operation warning", MessageBoxButtons.YesNo, MessageBoxIcon.Warning) == DialogResult.Yes)
            {
                try
                {
                    SySal.OperaDb.Schema.PC_REMOVE_CS_OR_BRICK.Call(((BrickTag)lvBricks.SelectedItems[0].Tag).IdInSet, ((BrickTag)lvBricks.SelectedItems[0].Tag).Set);
                    btnRefreshBrickSets_Click(sender, e);
                    GetBricksInBrickSet(lvBrickSets.SelectedItems[0].Tag.ToString());
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "Error removing brick set!", MessageBoxButtons.OK);
                }
            }
#else
            if (MessageBox.Show("Are you sure you want to delete the selected brick ?", "Operation warning", MessageBoxButtons.YesNo, MessageBoxIcon.Warning) == DialogResult.Yes)
            {
                try
                {
                    using (StreamWriter file_out2 = new StreamWriter("C:\\OperaBrickManager\\input2.txt"))
                    {

                        file_out2.WriteLine(((BrickTag)lvBricks.SelectedItems[0].Tag).Set + "\t" + ((BrickTag)lvBricks.SelectedItems[0].Tag).IdInSet);
                       file_out2.Flush();
                       file_out2.Close();
                       

                    }
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "No Write", MessageBoxButtons.OK);

                }
            }
#endif
        }

        private void btnExit_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void btnAddBrickSpace_Click(object sender, EventArgs e)
        {
            if (lvBrickSets.SelectedItems.Count != 1)
            {
                MessageBox.Show("A brick set must be selected", "Input warning", MessageBoxButtons.OK);
                return;
            }
            AddBrickSpaceForm ab = new AddBrickSpaceForm();
            ab.SetBrick = lvBrickSets.SelectedItems[0].Tag.ToString();

#if true
            if (ab.ShowDialog() == DialogResult.OK)
            {
                try
                {
                    SySal.OperaDb.Schema.PC_ADD_BRICK_SPACE.Call(ab.NewID, ab.SetBrick);
                    btnRefreshBrickSets_Click(sender, e);
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "Error inserting brick space!", MessageBoxButtons.OK);
 
                }

               
            }
#else
            if (ab.ShowDialog() == DialogResult.OK)
            {
                string test = ab.NewID + "\t"+ ab.SetBrick;
                MessageBox.Show(test);
                
               
            }
#endif
            ab.Dispose();

        }

        private void btnRemoveBrickSpace_Click(object sender, EventArgs e)
        {


            RemoveBrickSpaceForm ab = new RemoveBrickSpaceForm();
            ab.SetBrick = lvBrickSets.SelectedItems[0].Tag.ToString();


#if true  
            if (ab.ShowDialog() == DialogResult.OK)
            if (MessageBox.Show("Are you sure you want to delete the selected brick space?", "Operation warning", MessageBoxButtons.YesNo, MessageBoxIcon.Warning) == DialogResult.Yes)
            {
                try
                {
                    SySal.OperaDb.Schema.PC_REMOVE_BRICK_SPACE.Call(ab.OldIdSpace, ab.SetBrick);
                    

                }
                catch (Exception x)
                {
                    MessageBox.Show(x.Message, "Error removing brick space!", MessageBoxButtons.OK);
                }
            }

#else
            if (ab.ShowDialog() == DialogResult.OK)
            if (MessageBox.Show("Are you sure you want to delete the inserted brick space?", "Operation warning", MessageBoxButtons.YesNo, MessageBoxIcon.Warning) == DialogResult.Yes)
            {
                string test = System.Convert.ToString(ab.OldIdSpace) + "\t" + ab.SetBrick;
                MessageBox.Show(test);
                    
             
            }
#endif
        ab.Dispose();

               

        }

        private void btnShowAllBrick_Click(object sender, EventArgs e)
        {
            System.Data.DataSet ds = new DataSet();
            new OperaDbDataAdapter("Select id_Set, Id_brick, Minx, MaxX, MinY ,MaxY, MinZ, MaxZ, ZeroX, ZeroY, ZeroZ, id  from tb_eventbricks",
                DBConn).Fill(ds);
            lvBricks.Items.Clear();
            
            foreach (DataRow dr in ds.Tables[0].Rows)
            {
                ListViewItem lvi = lvBricks.Items.Add(dr[11].ToString());
                lvi.SubItems.Add(dr[0].ToString());
                lvi.SubItems.Add(dr[1].ToString());
                lvi.SubItems.Add(dr[2].ToString());
                lvi.SubItems.Add(dr[3].ToString());
                lvi.SubItems.Add(dr[4].ToString());
                lvi.SubItems.Add(dr[5].ToString());
                lvi.SubItems.Add(dr[6].ToString());
                lvi.SubItems.Add(dr[7].ToString());
                lvi.SubItems.Add(dr[8].ToString());
                lvi.SubItems.Add(dr[9].ToString());
                lvi.SubItems.Add(dr[10].ToString());
                lvi.Tag = new BrickTag(dr[0].ToString(), System.Convert.ToInt32(dr[1]));
            }

        }
        private void GetMarkInTableMarkSet(String Brick)
        {
/*
            int NumBrick;
            int NumBrickCs;
            String BrickCs;
            NumBrick = System.Convert.ToInt32(Brick);
            NumBrickCs = 3000000 + NumBrick;
            NumBrick = 1000000 + NumBrick;
            Brick = System.Convert.ToString(NumBrick);
            BrickCs = System.Convert.ToString(NumBrickCs);
 */
            System.Data.DataSet ds = new DataSet();

            new OperaDbDataAdapter("select * from tb_templatemarksets where id_eventbrick = " + Brick, DBConn).Fill(ds);
            lvMarkSet.Items.Clear();

            foreach (DataRow dr in ds.Tables[0].Rows)
            {
                ListViewItem lvi = lvMarkSet.Items.Add(dr[0].ToString());
                lvi.SubItems.Add(dr[1].ToString());
                lvi.SubItems.Add(dr[2].ToString());
                lvi.SubItems.Add(dr[3].ToString());
                lvi.SubItems.Add(dr[4].ToString());
                lvi.SubItems.Add(dr[5].ToString());
                lvi.SubItems.Add(dr[6].ToString());
                lvi.SubItems.Add(dr[7].ToString());
                lvi.SubItems.Add(dr[8].ToString());
                lvi.Tag = new BrickTag(dr[0].ToString(), System.Convert.ToInt32(dr[1]));

            }
        }

        private void btnInsertCSMark_Click(object sender, EventArgs e)
        {
            SySal.OperaDb.OperaDbTransaction trans = null;
            try
            {
                frmInsertMarkFromDB im = new frmInsertMarkFromDB();
                im.NumberBrick = lvBricks.SelectedItems[0].SubItems[0].Text;
                if (im.ShowDialog() == DialogResult.OK)
                {
                    trans = SySal.OperaDb.Schema.DB.BeginTransaction();
                    int nextmark = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT NVL(MAX(ID_MARK) + 1,0) FROM TB_TEMPLATEMARKSETS WHERE ID_EVENTBRICK = " + lvBricks.SelectedItems[0].SubItems[0].Text, SySal.OperaDb.Schema.DB, trans).ExecuteScalar());
                    new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_TEMPLATEMARKSETS (ID_EVENTBRICK, ID_MARK, POSX, POSY, MARKROW, MARKCOL, SHAPE, SIDE) (SELECT " + lvBricks.SelectedItems[0].SubItems[0].Text + ",ID_MARK + " + nextmark + ",POSX,POSY,MARKROW,MARKCOL,SHAPE,1 FROM TB_TEMPLATEMARKSETS WHERE ID_EVENTBRICK=" + im.CsBrick + ")", SySal.OperaDb.Schema.DB, trans).ExecuteNonQuery();
                    trans.Commit();
                    ListViewItem lvi = lvBricks.SelectedItems[0];
                    lvi.Selected = false;
                    lvi.Selected = true;
                }
            }
            catch (Exception x)
            {
                try
                {
                    if (trans != null) trans.Rollback();
                }
                catch (Exception) { }
                MessageBox.Show(x.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void btnLoadMarkFromFile_Click(object sender, EventArgs e)
        {
            SySal.OperaDb.OperaDbTransaction trans = null;
            try
            {
                frmLoadMarkFromFile lm = new frmLoadMarkFromFile();
                lm.NumberBrick = lvBricks.SelectedItems[0].SubItems[0].Text;
                if (lm.ShowDialog() == DialogResult.OK)
                {                    
                    trans = SySal.OperaDb.Schema.DB.BeginTransaction();
                    long nextmark = SySal.OperaDb.Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT NVL(MAX(ID_MARK),0) FROM TB_TEMPLATEMARKSETS WHERE ID_EVENTBRICK = " + lvBricks.SelectedItems[0].SubItems[0].Text, SySal.OperaDb.Schema.DB, trans).ExecuteScalar()) + 1;
                    foreach (SySal.BasicTypes.Vector v in lm.MarkSet.Marks)
                        SySal.OperaDb.Schema.TB_TEMPLATEMARKSETS.Insert(0L, System.Convert.ToInt64(lvBricks.SelectedItems[0].SubItems[0].Text), nextmark++, v.X, v.Y, 1, 1, lm.MarkSet.type.ToString(), (int)v.Z);                    
                    trans.Commit();
                    lvBricks.SelectedItems[0].Selected = true;
                }
            }
            catch (Exception x)
            {
                try
                {
                    if (trans != null) trans.Rollback();
                } 
                catch (Exception) {}
                MessageBox.Show(x.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void lvBricks_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (lvBricks.SelectedItems.Count != 1) return;
            GetMarkInTableMarkSet(lvBricks.SelectedItems[0].SubItems[0].Text);
        }

        private void lvBrickSets_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (lvBrickSets.SelectedItems.Count != 1) return;
            GetBricksInBrickSet(lvBrickSets.SelectedItems[0].Text);
        }      

    }
}