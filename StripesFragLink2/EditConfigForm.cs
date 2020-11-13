using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Processing.StripesFragLink2
{
	/// <summary>
	/// Form for configuration editing of <see cref="SySal.Processing.StripesFragLink2.StripesFragmentLinker">StripesFragmentLinker</see>.
	/// </summary>
	internal class EditConfigForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox MinGrains;
		private System.Windows.Forms.TextBox PosTol;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox SlopeTol;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.TextBox MergeSlopeTol;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.TextBox MergePosTol;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.TextBox MinSlope;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.Button OK;
		private System.Windows.Forms.Button Cancel;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		private System.Windows.Forms.TextBox SlopeTolIncreaseWithSlope;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.TrackBar MemorySavingBar;
		private System.Windows.Forms.CheckBox KeepLinkedTracksOnlyCheck;

		public StripesFragLink2.Configuration C;

		public EditConfigForm()
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
			this.label1 = new System.Windows.Forms.Label();
			this.MinGrains = new System.Windows.Forms.TextBox();
			this.PosTol = new System.Windows.Forms.TextBox();
			this.label2 = new System.Windows.Forms.Label();
			this.SlopeTol = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.MergeSlopeTol = new System.Windows.Forms.TextBox();
			this.label4 = new System.Windows.Forms.Label();
			this.MergePosTol = new System.Windows.Forms.TextBox();
			this.label5 = new System.Windows.Forms.Label();
			this.MinSlope = new System.Windows.Forms.TextBox();
			this.label6 = new System.Windows.Forms.Label();
			this.OK = new System.Windows.Forms.Button();
			this.Cancel = new System.Windows.Forms.Button();
			this.SlopeTolIncreaseWithSlope = new System.Windows.Forms.TextBox();
			this.label7 = new System.Windows.Forms.Label();
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.MemorySavingBar = new System.Windows.Forms.TrackBar();
			this.KeepLinkedTracksOnlyCheck = new System.Windows.Forms.CheckBox();
			this.groupBox1.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.MemorySavingBar)).BeginInit();
			this.SuspendLayout();
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(8, 8);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(72, 16);
			this.label1.TabIndex = 0;
			this.label1.Text = "Min Grains";
			this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// MinGrains
			// 
			this.MinGrains.Location = new System.Drawing.Point(80, 8);
			this.MinGrains.Name = "MinGrains";
			this.MinGrains.Size = new System.Drawing.Size(48, 20);
			this.MinGrains.TabIndex = 1;
			this.MinGrains.Text = "";
			// 
			// PosTol
			// 
			this.PosTol.Location = new System.Drawing.Point(264, 72);
			this.PosTol.Name = "PosTol";
			this.PosTol.Size = new System.Drawing.Size(48, 20);
			this.PosTol.TabIndex = 9;
			this.PosTol.Text = "";
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(8, 72);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(232, 16);
			this.label2.TabIndex = 8;
			this.label2.Text = "Pos Tolerance";
			this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// SlopeTol
			// 
			this.SlopeTol.Location = new System.Drawing.Point(264, 96);
			this.SlopeTol.Name = "SlopeTol";
			this.SlopeTol.Size = new System.Drawing.Size(48, 20);
			this.SlopeTol.TabIndex = 11;
			this.SlopeTol.Text = "";
			// 
			// label3
			// 
			this.label3.Location = new System.Drawing.Point(8, 96);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(232, 16);
			this.label3.TabIndex = 10;
			this.label3.Text = "Slope Tolerance";
			this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// MergeSlopeTol
			// 
			this.MergeSlopeTol.Location = new System.Drawing.Point(264, 32);
			this.MergeSlopeTol.Name = "MergeSlopeTol";
			this.MergeSlopeTol.Size = new System.Drawing.Size(48, 20);
			this.MergeSlopeTol.TabIndex = 7;
			this.MergeSlopeTol.Text = "";
			// 
			// label4
			// 
			this.label4.Location = new System.Drawing.Point(136, 32);
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size(120, 16);
			this.label4.TabIndex = 6;
			this.label4.Text = "Merge Slope Tol";
			this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// MergePosTol
			// 
			this.MergePosTol.Location = new System.Drawing.Point(264, 8);
			this.MergePosTol.Name = "MergePosTol";
			this.MergePosTol.Size = new System.Drawing.Size(48, 20);
			this.MergePosTol.TabIndex = 5;
			this.MergePosTol.Text = "";
			// 
			// label5
			// 
			this.label5.Location = new System.Drawing.Point(136, 8);
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size(120, 16);
			this.label5.TabIndex = 4;
			this.label5.Text = "Merge Pos Tol";
			this.label5.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// MinSlope
			// 
			this.MinSlope.Location = new System.Drawing.Point(80, 32);
			this.MinSlope.Name = "MinSlope";
			this.MinSlope.Size = new System.Drawing.Size(48, 20);
			this.MinSlope.TabIndex = 3;
			this.MinSlope.Text = "";
			// 
			// label6
			// 
			this.label6.Location = new System.Drawing.Point(8, 32);
			this.label6.Name = "label6";
			this.label6.Size = new System.Drawing.Size(72, 16);
			this.label6.TabIndex = 2;
			this.label6.Text = "Min Slope";
			this.label6.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// OK
			// 
			this.OK.Location = new System.Drawing.Point(248, 248);
			this.OK.Name = "OK";
			this.OK.Size = new System.Drawing.Size(64, 24);
			this.OK.TabIndex = 17;
			this.OK.Text = "OK";
			this.OK.Click += new System.EventHandler(this.OK_Click);
			// 
			// Cancel
			// 
			this.Cancel.Location = new System.Drawing.Point(8, 248);
			this.Cancel.Name = "Cancel";
			this.Cancel.Size = new System.Drawing.Size(64, 24);
			this.Cancel.TabIndex = 16;
			this.Cancel.Text = "Cancel";
			this.Cancel.Click += new System.EventHandler(this.Cancel_Click);
			// 
			// SlopeTolIncreaseWithSlope
			// 
			this.SlopeTolIncreaseWithSlope.Location = new System.Drawing.Point(264, 120);
			this.SlopeTolIncreaseWithSlope.Name = "SlopeTolIncreaseWithSlope";
			this.SlopeTolIncreaseWithSlope.Size = new System.Drawing.Size(48, 20);
			this.SlopeTolIncreaseWithSlope.TabIndex = 13;
			this.SlopeTolIncreaseWithSlope.Text = "";
			// 
			// label7
			// 
			this.label7.Location = new System.Drawing.Point(8, 120);
			this.label7.Name = "label7";
			this.label7.Size = new System.Drawing.Size(232, 16);
			this.label7.TabIndex = 12;
			this.label7.Text = "Slope Tolerance increases with Slope";
			this.label7.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// groupBox1
			// 
			this.groupBox1.Controls.AddRange(new System.Windows.Forms.Control[] {
																					this.MemorySavingBar});
			this.groupBox1.Location = new System.Drawing.Point(8, 152);
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Size = new System.Drawing.Size(304, 64);
			this.groupBox1.TabIndex = 14;
			this.groupBox1.TabStop = false;
			this.groupBox1.Text = "Memory saving";
			// 
			// MemorySavingBar
			// 
			this.MemorySavingBar.Location = new System.Drawing.Point(8, 16);
			this.MemorySavingBar.Maximum = 3;
			this.MemorySavingBar.Name = "MemorySavingBar";
			this.MemorySavingBar.Size = new System.Drawing.Size(288, 45);
			this.MemorySavingBar.TabIndex = 15;
			this.MemorySavingBar.Value = 1;
			// 
			// KeepLinkedTracksOnlyCheck
			// 
			this.KeepLinkedTracksOnlyCheck.Location = new System.Drawing.Point(8, 224);
			this.KeepLinkedTracksOnlyCheck.Name = "KeepLinkedTracksOnlyCheck";
			this.KeepLinkedTracksOnlyCheck.Size = new System.Drawing.Size(304, 24);
			this.KeepLinkedTracksOnlyCheck.TabIndex = 18;
			this.KeepLinkedTracksOnlyCheck.Text = "Keep linked tracks only";
			// 
			// EditConfigForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(320, 280);
			this.Controls.AddRange(new System.Windows.Forms.Control[] {
																		  this.KeepLinkedTracksOnlyCheck,
																		  this.groupBox1,
																		  this.SlopeTolIncreaseWithSlope,
																		  this.label7,
																		  this.Cancel,
																		  this.OK,
																		  this.MergeSlopeTol,
																		  this.label4,
																		  this.MergePosTol,
																		  this.label5,
																		  this.MinSlope,
																		  this.label6,
																		  this.SlopeTol,
																		  this.label3,
																		  this.PosTol,
																		  this.label2,
																		  this.MinGrains,
																		  this.label1});
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "EditConfigForm";
			this.Text = "EditConfigForm";
			this.Load += new System.EventHandler(this.EditConfigForm_Load);
			this.groupBox1.ResumeLayout(false);
			((System.ComponentModel.ISupportInitialize)(this.MemorySavingBar)).EndInit();
			this.ResumeLayout(false);

		}
		#endregion

		private void EditConfigForm_Load(object sender, System.EventArgs e)
		{
			KeepLinkedTracksOnlyCheck.Checked = C.KeepLinkedTracksOnly;
			MinGrains.Text = C.MinGrains.ToString();
			MinSlope.Text = C.MinSlope.ToString();
			PosTol.Text = C.PosTol.ToString();
			SlopeTol.Text = C.SlopeTol.ToString();
			SlopeTolIncreaseWithSlope.Text = C.SlopeTolIncreaseWithSlope.ToString();
			MergePosTol.Text = C.MergePosTol.ToString();
			MergeSlopeTol.Text = C.MergeSlopeTol.ToString();
			MemorySavingBar.Value = (int)C.MemorySaving;
		}

		private void OK_Click(object sender, System.EventArgs e)
		{
			try
			{
				C.KeepLinkedTracksOnly = KeepLinkedTracksOnlyCheck.Checked;
				C.MinGrains = Convert.ToInt32(MinGrains.Text);
				C.MinSlope = Convert.ToDouble(MinSlope.Text);
				C.PosTol = Convert.ToDouble(PosTol.Text);
				C.SlopeTol = Convert.ToDouble(SlopeTol.Text);
				C.SlopeTolIncreaseWithSlope = Convert.ToDouble(SlopeTolIncreaseWithSlope.Text);
				C.MergePosTol = Convert.ToDouble(MergePosTol.Text);
				C.MergeSlopeTol = Convert.ToDouble(MergeSlopeTol.Text);
				C.MemorySaving = (uint)MemorySavingBar.Value;
				this.DialogResult = DialogResult.OK;
				this.Close();
			}
			catch (System.Exception) {};
		}

		private void Cancel_Click(object sender, System.EventArgs e)
		{
			this.DialogResult = DialogResult.Cancel;
			this.Close();
		}
	}
}
