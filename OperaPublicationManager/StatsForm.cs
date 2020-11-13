using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Executables.OperaPublicationManager
{
	/// <summary>
	/// Shows execution statistics for a specified job.
	/// </summary>
	/// <remarks>
	/// <para>For all the tables involved in a certain job, this form shows three quantities (that are logged by the OPERAPUB procedures executing on the DB):
	/// <list type="bullet">
	/// <item><term>Total times</term><description>the total time spent (in seconds) is shown, together with a bar plot displaying the fraction spent on each table. This is especially useful to optimize DB indices and keys.</description></item>
	/// <item><term>Total rows</term><description>the total number of rows involved in the job (i.e. copied/compared/deleted) is shown, together with a bar plot displaying the fraction contributed by each table. This is especially useful for DB sizing computations.</description></item>
	/// <item><term>Average cost per row</term><description>the average time spent per each row in each table is shown. The total average time includes also overheads due to PL/SQL procedure processing, whereas the individual averages are computed using SQL statements only; the latter, 
	/// in turn, include also the SQL parsing time and SQL*Net TCP/IP round trips. While analyzing these results, one should keep in mind that the average cost per row can be very high for sparse data: e.g. for an INSERT that copies zero rows, parsing and round trips can take 2s, which is
	/// roughly the same time needed to copy 500 rows; hence, the average cost per row will be very different.</description></item>
	/// </list>
	/// </para>
	/// <para>The three <i>Export</i> buttons allow to export time data, row count data or row cost data to ASCII files.</para>
	/// <para>The <i>Refresh</i> button updates the statistics (if the related job is running).</para>
	/// </remarks>
	public class StatsForm : System.Windows.Forms.Form
	{
		private SySal.Controls.BackgroundPanel backgroundPanel1;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		private long JobId;

		private string DBLink;
		private SySal.Controls.Button btnRefresh;
		private SySal.Controls.GroupBox groupBox1;
		private SySal.Controls.GroupBox groupBox2;
		private SySal.Controls.GroupBox groupBox3;
		private SySal.Controls.PercentChart prchTime;
		private SySal.Controls.PercentChart prchRows;
		private SySal.Controls.PercentChart prchCost;
		private SySal.Controls.StaticText staticText1;
		private System.Windows.Forms.TextBox textTotalTime;
		private System.Windows.Forms.TextBox textTotalRows;
		private SySal.Controls.StaticText staticText2;
		private SySal.Controls.Button btnExportTimeChart;
		private SySal.Controls.Button btnExportRowChart;
		private SySal.Controls.Button btnExportCostChart;
		private System.Windows.Forms.TextBox textAvgCost;
		private SySal.Controls.StaticText staticText3;

		private SySal.OperaDb.OperaDbConnection DBConn;

		/// <summary>
		/// Creates a new StatsForm.
		/// </summary>
		public StatsForm()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//
			// TODO: Add any constructor code after InitializeComponent call
			//
		}

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		protected override void Dispose( bool disposing )
		{
			if( disposing )
			{
				if(components != null)
				{
					components.Dispose();
				}
			}
			base.Dispose( disposing );
		}

		#region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.backgroundPanel1 = new SySal.Controls.BackgroundPanel();
			this.prchTime = new SySal.Controls.PercentChart();
			this.prchRows = new SySal.Controls.PercentChart();
			this.prchCost = new SySal.Controls.PercentChart();
			this.btnRefresh = new SySal.Controls.Button();
			this.groupBox1 = new SySal.Controls.GroupBox();
			this.groupBox2 = new SySal.Controls.GroupBox();
			this.groupBox3 = new SySal.Controls.GroupBox();
			this.staticText1 = new SySal.Controls.StaticText();
			this.textTotalTime = new System.Windows.Forms.TextBox();
			this.textTotalRows = new System.Windows.Forms.TextBox();
			this.staticText2 = new SySal.Controls.StaticText();
			this.btnExportTimeChart = new SySal.Controls.Button();
			this.btnExportRowChart = new SySal.Controls.Button();
			this.btnExportCostChart = new SySal.Controls.Button();
			this.textAvgCost = new System.Windows.Forms.TextBox();
			this.staticText3 = new SySal.Controls.StaticText();
			this.SuspendLayout();
			// 
			// backgroundPanel1
			// 
			this.backgroundPanel1.BackColor = System.Drawing.Color.White;
			this.backgroundPanel1.Location = new System.Drawing.Point(0, 0);
			this.backgroundPanel1.Name = "backgroundPanel1";
			this.backgroundPanel1.Size = new System.Drawing.Size(1064, 544);
			this.backgroundPanel1.TabIndex = 10;
			// 
			// prchTime
			// 
			this.prchTime.BackColor = System.Drawing.Color.White;
			this.prchTime.Location = new System.Drawing.Point(16, 40);
			this.prchTime.Name = "prchTime";
			this.prchTime.Size = new System.Drawing.Size(504, 192);
			this.prchTime.TabIndex = 11;
			// 
			// prchRows
			// 
			this.prchRows.BackColor = System.Drawing.Color.White;
			this.prchRows.Location = new System.Drawing.Point(16, 304);
			this.prchRows.Name = "prchRows";
			this.prchRows.Size = new System.Drawing.Size(504, 192);
			this.prchRows.TabIndex = 12;
			// 
			// prchCost
			// 
			this.prchCost.BackColor = System.Drawing.Color.White;
			this.prchCost.Location = new System.Drawing.Point(544, 40);
			this.prchCost.Name = "prchCost";
			this.prchCost.Size = new System.Drawing.Size(504, 312);
			this.prchCost.TabIndex = 13;
			// 
			// btnRefresh
			// 
			this.btnRefresh.BackColor = System.Drawing.Color.White;
			this.btnRefresh.ButtonText = "Refresh";
			this.btnRefresh.Location = new System.Drawing.Point(952, 504);
			this.btnRefresh.Name = "btnRefresh";
			this.btnRefresh.Size = new System.Drawing.Size(104, 32);
			this.btnRefresh.TabIndex = 14;
			this.btnRefresh.Click += new System.EventHandler(this.OnRefresh);
			this.btnRefresh.DoubleClick += new System.EventHandler(this.OnRefresh);
			// 
			// groupBox1
			// 
			this.groupBox1.BackColor = System.Drawing.Color.White;
			this.groupBox1.ClosedPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
			this.groupBox1.IsOpen = true;
			this.groupBox1.IsStatic = true;
			this.groupBox1.LabelText = "Time spent (s)";
			this.groupBox1.Location = new System.Drawing.Point(8, 8);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.OpenPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
			this.groupBox1.Size = new System.Drawing.Size(520, 264);
			this.groupBox1.TabIndex = 15;
			// 
			// groupBox2
			// 
			this.groupBox2.BackColor = System.Drawing.Color.White;
			this.groupBox2.ClosedPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
			this.groupBox2.IsOpen = true;
			this.groupBox2.IsStatic = true;
			this.groupBox2.LabelText = "Rows";
			this.groupBox2.Location = new System.Drawing.Point(8, 272);
			this.groupBox2.Name = "groupBox2";
			this.groupBox2.OpenPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
			this.groupBox2.Size = new System.Drawing.Size(520, 264);
			this.groupBox2.TabIndex = 16;
			// 
			// groupBox3
			// 
			this.groupBox3.BackColor = System.Drawing.Color.White;
			this.groupBox3.ClosedPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
			this.groupBox3.IsOpen = true;
			this.groupBox3.IsStatic = true;
			this.groupBox3.LabelText = "Cost (µs/row)";
			this.groupBox3.Location = new System.Drawing.Point(536, 8);
			this.groupBox3.Name = "groupBox3";
			this.groupBox3.OpenPosition = new System.Drawing.Rectangle(0, 0, 0, 0);
			this.groupBox3.Size = new System.Drawing.Size(520, 384);
			this.groupBox3.TabIndex = 17;
			// 
			// staticText1
			// 
			this.staticText1.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText1.LabelText = "Total time (s)";
			this.staticText1.Location = new System.Drawing.Point(16, 240);
			this.staticText1.Name = "staticText1";
			this.staticText1.Size = new System.Drawing.Size(96, 24);
			this.staticText1.TabIndex = 18;
			this.staticText1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// textTotalTime
			// 
			this.textTotalTime.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(255)), ((System.Byte)(255)));
			this.textTotalTime.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textTotalTime.ForeColor = System.Drawing.Color.Navy;
			this.textTotalTime.Location = new System.Drawing.Point(120, 240);
			this.textTotalTime.Name = "textTotalTime";
			this.textTotalTime.ReadOnly = true;
			this.textTotalTime.Size = new System.Drawing.Size(80, 20);
			this.textTotalTime.TabIndex = 19;
			this.textTotalTime.Text = "";
			// 
			// textTotalRows
			// 
			this.textTotalRows.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(255)), ((System.Byte)(255)));
			this.textTotalRows.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textTotalRows.ForeColor = System.Drawing.Color.Navy;
			this.textTotalRows.Location = new System.Drawing.Point(120, 504);
			this.textTotalRows.Name = "textTotalRows";
			this.textTotalRows.ReadOnly = true;
			this.textTotalRows.Size = new System.Drawing.Size(80, 20);
			this.textTotalRows.TabIndex = 21;
			this.textTotalRows.Text = "";
			// 
			// staticText2
			// 
			this.staticText2.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText2.LabelText = "Total rows";
			this.staticText2.Location = new System.Drawing.Point(16, 504);
			this.staticText2.Name = "staticText2";
			this.staticText2.Size = new System.Drawing.Size(96, 24);
			this.staticText2.TabIndex = 20;
			this.staticText2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// btnExportTimeChart
			// 
			this.btnExportTimeChart.BackColor = System.Drawing.Color.White;
			this.btnExportTimeChart.ButtonText = "Export time chart";
			this.btnExportTimeChart.Location = new System.Drawing.Point(536, 408);
			this.btnExportTimeChart.Name = "btnExportTimeChart";
			this.btnExportTimeChart.Size = new System.Drawing.Size(144, 32);
			this.btnExportTimeChart.TabIndex = 22;
			this.btnExportTimeChart.Click += new System.EventHandler(this.btnExportTimeChart_Click);
			this.btnExportTimeChart.DoubleClick += new System.EventHandler(this.btnExportTimeChart_Click);
			// 
			// btnExportRowChart
			// 
			this.btnExportRowChart.BackColor = System.Drawing.Color.White;
			this.btnExportRowChart.ButtonText = "Export row chart";
			this.btnExportRowChart.Location = new System.Drawing.Point(536, 456);
			this.btnExportRowChart.Name = "btnExportRowChart";
			this.btnExportRowChart.Size = new System.Drawing.Size(144, 32);
			this.btnExportRowChart.TabIndex = 23;
			this.btnExportRowChart.Click += new System.EventHandler(this.btnExportRowChart_Click);
			this.btnExportRowChart.DoubleClick += new System.EventHandler(this.btnExportRowChart_Click);
			// 
			// btnExportCostChart
			// 
			this.btnExportCostChart.BackColor = System.Drawing.Color.White;
			this.btnExportCostChart.ButtonText = "Export cost chart";
			this.btnExportCostChart.Location = new System.Drawing.Point(536, 504);
			this.btnExportCostChart.Name = "btnExportCostChart";
			this.btnExportCostChart.Size = new System.Drawing.Size(144, 32);
			this.btnExportCostChart.TabIndex = 24;
			this.btnExportCostChart.Click += new System.EventHandler(this.btnExportCostChart_Click);
			this.btnExportCostChart.DoubleClick += new System.EventHandler(this.btnExportCostChart_Click);
			// 
			// textAvgCost
			// 
			this.textAvgCost.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(192)), ((System.Byte)(255)), ((System.Byte)(255)));
			this.textAvgCost.Font = new System.Drawing.Font("Arial", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.textAvgCost.ForeColor = System.Drawing.Color.Navy;
			this.textAvgCost.Location = new System.Drawing.Point(744, 360);
			this.textAvgCost.Name = "textAvgCost";
			this.textAvgCost.ReadOnly = true;
			this.textAvgCost.Size = new System.Drawing.Size(80, 20);
			this.textAvgCost.TabIndex = 26;
			this.textAvgCost.Text = "";
			// 
			// staticText3
			// 
			this.staticText3.BackColor = System.Drawing.Color.FromArgb(((System.Byte)(126)), ((System.Byte)(181)), ((System.Byte)(232)));
			this.staticText3.LabelText = "Average cost (µs/row)";
			this.staticText3.Location = new System.Drawing.Point(544, 360);
			this.staticText3.Name = "staticText3";
			this.staticText3.Size = new System.Drawing.Size(168, 24);
			this.staticText3.TabIndex = 27;
			this.staticText3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// StatsForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(1064, 544);
			this.Controls.Add(this.staticText3);
			this.Controls.Add(this.textAvgCost);
			this.Controls.Add(this.btnExportCostChart);
			this.Controls.Add(this.btnExportRowChart);
			this.Controls.Add(this.btnExportTimeChart);
			this.Controls.Add(this.textTotalRows);
			this.Controls.Add(this.staticText2);
			this.Controls.Add(this.textTotalTime);
			this.Controls.Add(this.staticText1);
			this.Controls.Add(this.prchCost);
			this.Controls.Add(this.groupBox3);
			this.Controls.Add(this.prchRows);
			this.Controls.Add(this.groupBox2);
			this.Controls.Add(this.prchTime);
			this.Controls.Add(this.groupBox1);
			this.Controls.Add(this.btnRefresh);
			this.Controls.Add(this.backgroundPanel1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "StatsForm";
			this.Text = "Publication Statistics";
			this.ResumeLayout(false);

		}
		#endregion

		/// <summary>
		/// Shows the dialog, filling the plots with statistics from the specified job.
		/// </summary>
		/// <param name="jobid">the ID of the job for which statistics are needed.</param>
		/// <param name="dbconn">the DB connection to be used.</param>
		internal void ShowDialog(long jobid, SySal.OperaDb.OperaDbConnection dbconn)
		{			
			JobId = jobid;
			DBConn = dbconn;
			OnRefresh(null, null);
			ShowDialog();
		}

		private void OnRefresh(object sender, System.EventArgs e)
		{
			string wherestr = "";
			if (JobId > 0)
			{
				wherestr = " WHERE ID_JOB = " + JobId;
				this.Text = "Publication Statistics for Job #" + JobId;
			}
			else this.Text = "Publication Statistics for all jobs";
			System.Data.DataSet ds = new System.Data.DataSet();
			new SySal.OperaDb.OperaDbDataAdapter("SELECT ACTIVITY, SUM(SECONDS) as SECONDS, SUM(ROWS_AFFECTED) as ROWS_AFFECTED FROM PT_JOBSTATS" + wherestr + " GROUP BY ACTIVITY ORDER BY SECONDS DESC", DBConn, null).Fill(ds);
			double totalsecs = 0.0;
			long totalrows = 0;
			double maxcost = 0.0;
			foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
			{
				totalsecs += SySal.OperaDb.Convert.ToDouble(dr[1]);
				totalrows += SySal.OperaDb.Convert.ToInt64(dr[2]);
				if (SySal.OperaDb.Convert.ToInt64(dr[2]) > 0)
				{
					double cost = SySal.OperaDb.Convert.ToDouble(dr[1]) / SySal.OperaDb.Convert.ToInt64(dr[2]) * 1e6;
					if (cost > maxcost) maxcost = cost;
				}
			}
			textTotalTime.Text = totalsecs.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
			textTotalRows.Text = totalrows.ToString();
			if (totalrows > 0)
				textAvgCost.Text = (totalsecs / totalrows * 1e6).ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
			else 
				textAvgCost.Text = "";
			int i;
			prchTime.Items.Clear();
			prchRows.Items.Clear();
			prchCost.Items.Clear();
			for (i = 0; i < ds.Tables[0].Rows.Count; i++)
			{
				SySal.Controls.PercentChartItem pcit = null;
				SySal.Controls.PercentChartItem pcir = null;
				SySal.Controls.PercentChartItem pcic = null;
				if (totalsecs == 0.0)
					pcit = new SySal.Controls.PercentChartItem(ds.Tables[0].Rows[i][0].ToString(), 0.0);
				else
				{
					double percent = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[i][1]) / totalsecs;
					pcit = new SySal.Controls.PercentChartItem(ds.Tables[0].Rows[i][0].ToString() + "(" + (percent * 100.0).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "%)", percent);
				}	
				prchTime.Items.Add(pcit);
				if (totalrows == 0)
					pcir = new SySal.Controls.PercentChartItem(ds.Tables[0].Rows[i][0].ToString(), 0);
				else
				{
					double percent = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[i][2]) / totalrows;
					pcir = new SySal.Controls.PercentChartItem(ds.Tables[0].Rows[i][0].ToString() + "(" + (percent * 100.0).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "%)", percent);
				}	
				prchRows.Items.Add(pcir);
				if (maxcost > 0.0)
				{
					if (SySal.OperaDb.Convert.ToInt64(ds.Tables[0].Rows[i][2]) > 0)
					{
						double cost = SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[i][1]) / SySal.OperaDb.Convert.ToDouble(ds.Tables[0].Rows[i][2]) * 1e6;
						double percent = cost / maxcost;
						pcic = new SySal.Controls.PercentChartItem(ds.Tables[0].Rows[i][0].ToString() + "(" + cost.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + ")", percent);
						prchCost.Items.Add(pcic);
					}
				}	
			}		
		}

		private void btnExportCostChart_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sdlg = new SaveFileDialog();
			sdlg.Title = "Select export file";
			sdlg.Filter = "Tab-delimited Text files (*.txt)|*.txt|All files (*.*)|*.*";
			if (sdlg.ShowDialog() == DialogResult.OK) ExportToFile(prchCost.Items, sdlg.FileName);								
		}

		private void btnExportRowChart_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sdlg = new SaveFileDialog();
			sdlg.Title = "Select export file";
			sdlg.Filter = "Tab-delimited Text files (*.txt)|*.txt|All files (*.*)|*.*";
			if (sdlg.ShowDialog() == DialogResult.OK) ExportToFile(prchRows.Items, sdlg.FileName);										
		}

		private void btnExportTimeChart_Click(object sender, System.EventArgs e)
		{
			SaveFileDialog sdlg = new SaveFileDialog();
			sdlg.Title = "Select export file";
			sdlg.Filter = "Tab-delimited Text files (*.txt)|*.txt|All files (*.*)|*.*";
			if (sdlg.ShowDialog() == DialogResult.OK) ExportToFile(prchTime.Items, sdlg.FileName);										
		}

		private static void ExportToFile(SySal.Controls.PercentChartItemCollection items, string fname)
		{
			System.IO.StreamWriter wr = null;
			try
			{
				wr = new System.IO.StreamWriter(fname);
				int i;
				for (i = 0; i < items.Count; i++)
				{
					if (i > 0) wr.WriteLine();
					wr.Write(items[i].Text + "\t" + items[i].Percent);
				}
			}
			catch(Exception x)
			{
				MessageBox.Show(x.Message, "Error exporting chart to file", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
			if (wr != null) wr.Close();			
		}
	}
}
