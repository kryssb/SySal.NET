using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Processing.SmartTracking
{
	internal class EditConfigForm : System.Windows.Forms.Form
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox MinSlope;
		private System.Windows.Forms.TextBox MaxSlope;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.TextBox CellOverflow;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.TextBox MinGrains0;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox MinGrainsHorizontal;
		private System.Windows.Forms.TextBox MaxArea;
		private System.Windows.Forms.Label label8;
		private System.Windows.Forms.TextBox MinArea;
		private System.Windows.Forms.Label label9;
		private System.Windows.Forms.ColumnHeader columnHeader1;
		private System.Windows.Forms.ColumnHeader columnHeader2;
		private System.Windows.Forms.ColumnHeader columnHeader3;
		private System.Windows.Forms.ListView TriggerList;
		private System.Windows.Forms.TextBox MinGrains01;
		private System.Windows.Forms.TextBox XYAlignmentTolerance;
		private System.Windows.Forms.Panel GrainsDisplay;
		private System.Windows.Forms.Button NewTriggerButton;
		private System.Windows.Forms.Button DelTriggerButton;
		private System.Windows.Forms.Button OKButton;
		private System.Windows.Forms.Button MyCancelButton;
		private System.Windows.Forms.TextBox CellNumX;
		private System.Windows.Forms.Label label10;
        private TextBox CellNumY;
        private TextBox ReplicaRadius;
        private Label label11;
        private TextBox MinReplicas;
        private Label label12;
        private TextBox ReplicaSampleDivider;
        private Label label13;
        private CheckBox AllowOverlap;
        private TextBox DeltaZMultiplier;
        private Label label6;
        private Label label7;
        private TextBox MaxProcessors;
        private TextBox MaxTrackingTime;
        private Label label14;
        private TextBox InitialMultiplicity;
        private Label label15;
		private Configuration C;

		public Configuration Config
		{
			get
			{
				return (Configuration)C.Clone();
			}

			set
			{
				C = (Configuration)(value.Clone());
				TriggerList.Items.Clear();
				foreach (Configuration.TriggerInfo trg in C.Triggers)
				{
					TriggerList.Items.Add(trg.TopLayer.ToString());
					TriggerList.Items[TriggerList.Items.Count - 1].SubItems.Add(trg.BottomLayer.ToString());
					string trgtext = "";
					foreach (uint tl in trg.TriggerLayers)
					{
						if (trgtext.Length > 0)
							trgtext += ",";
						trgtext += tl.ToString();
					}
					TriggerList.Items[TriggerList.Items.Count - 1].SubItems.Add(trgtext);
				}
				MinSlope.Text = C.MinSlope.ToString();
				MaxSlope.Text = C.MaxSlope.ToString();
				MinArea.Text = C.MinArea.ToString();
				MaxArea.Text = C.MaxArea.ToString();
				XYAlignmentTolerance.Text = C.AlignTol.ToString();
				CellOverflow.Text = C.CellOverflow.ToString();
				CellNumX.Text = C.CellNumX.ToString();
                CellNumY.Text = C.CellNumY.ToString();
                InitialMultiplicity.Text = C.InitialMultiplicity.ToString();
				MinGrains0.Text = C.MinGrainsForVerticalTrack.ToString();
				MinGrains01.Text = C.MinGrainsSlope01.ToString();
				MinGrainsHorizontal.Text = C.MinGrainsForHorizontalTrack.ToString();
                MinReplicas.Text = C.MinReplicas.ToString();
                ReplicaRadius.Text = C.ReplicaRadius.ToString();
                ReplicaSampleDivider.Text = C.ReplicaSampleDivider.ToString();
                DeltaZMultiplier.Text = C.DeltaZMultiplier.ToString();
                MaxProcessors.Text = C.MaxProcessors.ToString();
                MaxTrackingTime.Text = C.MaxTrackingTimeMS.ToString();
                AllowOverlap.Checked = C.AllowOverlap;
			}
		}

		internal EditConfigForm()
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(EditConfigForm));
            this.label1 = new System.Windows.Forms.Label();
            this.MinSlope = new System.Windows.Forms.TextBox();
            this.MaxSlope = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.XYAlignmentTolerance = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.CellOverflow = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.MinGrains0 = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.MinGrains01 = new System.Windows.Forms.TextBox();
            this.MinGrainsHorizontal = new System.Windows.Forms.TextBox();
            this.MaxArea = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.MinArea = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.GrainsDisplay = new System.Windows.Forms.Panel();
            this.TriggerList = new System.Windows.Forms.ListView();
            this.columnHeader1 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader2 = new System.Windows.Forms.ColumnHeader();
            this.columnHeader3 = new System.Windows.Forms.ColumnHeader();
            this.NewTriggerButton = new System.Windows.Forms.Button();
            this.DelTriggerButton = new System.Windows.Forms.Button();
            this.OKButton = new System.Windows.Forms.Button();
            this.MyCancelButton = new System.Windows.Forms.Button();
            this.CellNumX = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            this.CellNumY = new System.Windows.Forms.TextBox();
            this.ReplicaRadius = new System.Windows.Forms.TextBox();
            this.label11 = new System.Windows.Forms.Label();
            this.MinReplicas = new System.Windows.Forms.TextBox();
            this.label12 = new System.Windows.Forms.Label();
            this.ReplicaSampleDivider = new System.Windows.Forms.TextBox();
            this.label13 = new System.Windows.Forms.Label();
            this.AllowOverlap = new System.Windows.Forms.CheckBox();
            this.DeltaZMultiplier = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.MaxProcessors = new System.Windows.Forms.TextBox();
            this.MaxTrackingTime = new System.Windows.Forms.TextBox();
            this.label14 = new System.Windows.Forms.Label();
            this.InitialMultiplicity = new System.Windows.Forms.TextBox();
            this.label15 = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.Location = new System.Drawing.Point(8, 8);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(72, 24);
            this.label1.TabIndex = 0;
            this.label1.Text = "Min. Slope";
            this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // MinSlope
            // 
            this.MinSlope.Location = new System.Drawing.Point(88, 8);
            this.MinSlope.Name = "MinSlope";
            this.MinSlope.Size = new System.Drawing.Size(48, 20);
            this.MinSlope.TabIndex = 1;
            this.MinSlope.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.MinSlope.Validating += new System.ComponentModel.CancelEventHandler(this.MinSlope_Validate);
            // 
            // MaxSlope
            // 
            this.MaxSlope.Location = new System.Drawing.Point(88, 32);
            this.MaxSlope.Name = "MaxSlope";
            this.MaxSlope.Size = new System.Drawing.Size(48, 20);
            this.MaxSlope.TabIndex = 3;
            this.MaxSlope.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.MaxSlope.Validating += new System.ComponentModel.CancelEventHandler(this.MaxSlope_Validate);
            // 
            // label2
            // 
            this.label2.Location = new System.Drawing.Point(8, 32);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(72, 24);
            this.label2.TabIndex = 2;
            this.label2.Text = "Max. Slope";
            this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // XYAlignmentTolerance
            // 
            this.XYAlignmentTolerance.Location = new System.Drawing.Point(88, 131);
            this.XYAlignmentTolerance.Name = "XYAlignmentTolerance";
            this.XYAlignmentTolerance.Size = new System.Drawing.Size(48, 20);
            this.XYAlignmentTolerance.TabIndex = 9;
            this.XYAlignmentTolerance.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.XYAlignmentTolerance.Validating += new System.ComponentModel.CancelEventHandler(this.XYAlignmentTolerance_Validate);
            // 
            // label3
            // 
            this.label3.Location = new System.Drawing.Point(9, 129);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(73, 24);
            this.label3.TabIndex = 8;
            this.label3.Text = "XY Align. Tol.";
            this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // CellOverflow
            // 
            this.CellOverflow.Location = new System.Drawing.Point(88, 56);
            this.CellOverflow.Name = "CellOverflow";
            this.CellOverflow.Size = new System.Drawing.Size(48, 20);
            this.CellOverflow.TabIndex = 11;
            this.CellOverflow.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.CellOverflow.Validating += new System.ComponentModel.CancelEventHandler(this.CellOverflow_Validate);
            // 
            // label4
            // 
            this.label4.Location = new System.Drawing.Point(8, 54);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(72, 24);
            this.label4.TabIndex = 10;
            this.label4.Text = "Cell Overflow";
            this.label4.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // MinGrains0
            // 
            this.MinGrains0.Location = new System.Drawing.Point(172, 326);
            this.MinGrains0.Name = "MinGrains0";
            this.MinGrains0.Size = new System.Drawing.Size(48, 20);
            this.MinGrains0.TabIndex = 15;
            this.MinGrains0.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.MinGrains0.Validating += new System.ComponentModel.CancelEventHandler(this.MinGrains0_Validate);
            // 
            // label5
            // 
            this.label5.Location = new System.Drawing.Point(8, 325);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(161, 24);
            this.label5.TabIndex = 14;
            this.label5.Text = "Min Grains at slope = 0, 0.1, Inf";
            this.label5.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // MinGrains01
            // 
            this.MinGrains01.Location = new System.Drawing.Point(222, 326);
            this.MinGrains01.Name = "MinGrains01";
            this.MinGrains01.Size = new System.Drawing.Size(48, 20);
            this.MinGrains01.TabIndex = 17;
            this.MinGrains01.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.MinGrains01.Validating += new System.ComponentModel.CancelEventHandler(this.MinGrains01_Validate);
            // 
            // MinGrainsHorizontal
            // 
            this.MinGrainsHorizontal.Location = new System.Drawing.Point(272, 326);
            this.MinGrainsHorizontal.Name = "MinGrainsHorizontal";
            this.MinGrainsHorizontal.Size = new System.Drawing.Size(48, 20);
            this.MinGrainsHorizontal.TabIndex = 19;
            this.MinGrainsHorizontal.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.MinGrainsHorizontal.Validating += new System.ComponentModel.CancelEventHandler(this.MinGrainsHorizontal_Validate);
            // 
            // MaxArea
            // 
            this.MaxArea.Location = new System.Drawing.Point(272, 32);
            this.MaxArea.Name = "MaxArea";
            this.MaxArea.Size = new System.Drawing.Size(48, 20);
            this.MaxArea.TabIndex = 7;
            this.MaxArea.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.MaxArea.Validating += new System.ComponentModel.CancelEventHandler(this.MaxArea_Validate);
            // 
            // label8
            // 
            this.label8.Location = new System.Drawing.Point(143, 32);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(104, 24);
            this.label8.TabIndex = 6;
            this.label8.Text = "Max. Area (pixels)";
            this.label8.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // MinArea
            // 
            this.MinArea.Location = new System.Drawing.Point(272, 8);
            this.MinArea.Name = "MinArea";
            this.MinArea.Size = new System.Drawing.Size(48, 20);
            this.MinArea.TabIndex = 5;
            this.MinArea.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.MinArea.Validating += new System.ComponentModel.CancelEventHandler(this.MinArea_Validate);
            // 
            // label9
            // 
            this.label9.Location = new System.Drawing.Point(143, 8);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(104, 24);
            this.label9.TabIndex = 4;
            this.label9.Text = "Min. Area (pixels)";
            this.label9.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // GrainsDisplay
            // 
            this.GrainsDisplay.BackColor = System.Drawing.Color.White;
            this.GrainsDisplay.BorderStyle = System.Windows.Forms.BorderStyle.Fixed3D;
            this.GrainsDisplay.Location = new System.Drawing.Point(328, 8);
            this.GrainsDisplay.Name = "GrainsDisplay";
            this.GrainsDisplay.Size = new System.Drawing.Size(320, 368);
            this.GrainsDisplay.TabIndex = 23;
            this.GrainsDisplay.Paint += new System.Windows.Forms.PaintEventHandler(this.CurvePaint);
            // 
            // TriggerList
            // 
            this.TriggerList.BackColor = System.Drawing.SystemColors.Window;
            this.TriggerList.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.columnHeader1,
            this.columnHeader2,
            this.columnHeader3});
            this.TriggerList.FullRowSelect = true;
            this.TriggerList.GridLines = true;
            this.TriggerList.Location = new System.Drawing.Point(8, 184);
            this.TriggerList.MultiSelect = false;
            this.TriggerList.Name = "TriggerList";
            this.TriggerList.Size = new System.Drawing.Size(312, 104);
            this.TriggerList.TabIndex = 20;
            this.TriggerList.UseCompatibleStateImageBehavior = false;
            this.TriggerList.View = System.Windows.Forms.View.Details;
            // 
            // columnHeader1
            // 
            this.columnHeader1.Text = "Top Layer";
            this.columnHeader1.Width = 74;
            // 
            // columnHeader2
            // 
            this.columnHeader2.Text = "Bottom Layer";
            this.columnHeader2.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader2.Width = 86;
            // 
            // columnHeader3
            // 
            this.columnHeader3.Text = "Trigger Layers";
            this.columnHeader3.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.columnHeader3.Width = 148;
            // 
            // NewTriggerButton
            // 
            this.NewTriggerButton.Location = new System.Drawing.Point(8, 296);
            this.NewTriggerButton.Name = "NewTriggerButton";
            this.NewTriggerButton.Size = new System.Drawing.Size(144, 24);
            this.NewTriggerButton.TabIndex = 21;
            this.NewTriggerButton.Text = "New Trigger";
            this.NewTriggerButton.Click += new System.EventHandler(this.NewTriggerButton_Click);
            // 
            // DelTriggerButton
            // 
            this.DelTriggerButton.Location = new System.Drawing.Point(176, 296);
            this.DelTriggerButton.Name = "DelTriggerButton";
            this.DelTriggerButton.Size = new System.Drawing.Size(144, 24);
            this.DelTriggerButton.TabIndex = 22;
            this.DelTriggerButton.Text = "Delete Trigger";
            this.DelTriggerButton.Click += new System.EventHandler(this.DelTriggerButton_Click);
            // 
            // OKButton
            // 
            this.OKButton.BackColor = System.Drawing.SystemColors.Control;
            this.OKButton.Location = new System.Drawing.Point(576, 384);
            this.OKButton.Name = "OKButton";
            this.OKButton.Size = new System.Drawing.Size(72, 24);
            this.OKButton.TabIndex = 25;
            this.OKButton.Text = "OK";
            this.OKButton.UseVisualStyleBackColor = false;
            this.OKButton.Click += new System.EventHandler(this.OKButton_Click);
            // 
            // MyCancelButton
            // 
            this.MyCancelButton.Location = new System.Drawing.Point(8, 384);
            this.MyCancelButton.Name = "MyCancelButton";
            this.MyCancelButton.Size = new System.Drawing.Size(72, 24);
            this.MyCancelButton.TabIndex = 24;
            this.MyCancelButton.Text = "Cancel";
            this.MyCancelButton.Click += new System.EventHandler(this.MyCancelButton_Click);
            // 
            // CellNumX
            // 
            this.CellNumX.Location = new System.Drawing.Point(222, 56);
            this.CellNumX.Name = "CellNumX";
            this.CellNumX.Size = new System.Drawing.Size(48, 20);
            this.CellNumX.TabIndex = 13;
            this.CellNumX.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.CellNumX.Validating += new System.ComponentModel.CancelEventHandler(this.CellNumX_Validate);
            // 
            // label10
            // 
            this.label10.Location = new System.Drawing.Point(143, 54);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(75, 24);
            this.label10.TabIndex = 12;
            this.label10.Text = "Cell Num. X/Y";
            this.label10.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // CellNumY
            // 
            this.CellNumY.Location = new System.Drawing.Point(272, 56);
            this.CellNumY.Name = "CellNumY";
            this.CellNumY.Size = new System.Drawing.Size(48, 20);
            this.CellNumY.TabIndex = 27;
            this.CellNumY.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.CellNumY.Validating += new System.ComponentModel.CancelEventHandler(this.CellNumY_Validate);
            // 
            // ReplicaRadius
            // 
            this.ReplicaRadius.Location = new System.Drawing.Point(272, 83);
            this.ReplicaRadius.Name = "ReplicaRadius";
            this.ReplicaRadius.Size = new System.Drawing.Size(48, 20);
            this.ReplicaRadius.TabIndex = 29;
            this.ReplicaRadius.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ReplicaRadius.Validating += new System.ComponentModel.CancelEventHandler(this.ReplicaRadius_Validate);
            // 
            // label11
            // 
            this.label11.Location = new System.Drawing.Point(142, 81);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(81, 24);
            this.label11.TabIndex = 28;
            this.label11.Text = "Replica Radius";
            this.label11.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // MinReplicas
            // 
            this.MinReplicas.Location = new System.Drawing.Point(88, 83);
            this.MinReplicas.Name = "MinReplicas";
            this.MinReplicas.Size = new System.Drawing.Size(48, 20);
            this.MinReplicas.TabIndex = 31;
            this.MinReplicas.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.MinReplicas.Validating += new System.ComponentModel.CancelEventHandler(this.MinReplicas_Validate);
            // 
            // label12
            // 
            this.label12.Location = new System.Drawing.Point(8, 83);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(72, 24);
            this.label12.TabIndex = 30;
            this.label12.Text = "Min. Replicas";
            this.label12.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // ReplicaSampleDivider
            // 
            this.ReplicaSampleDivider.Location = new System.Drawing.Point(272, 106);
            this.ReplicaSampleDivider.Name = "ReplicaSampleDivider";
            this.ReplicaSampleDivider.Size = new System.Drawing.Size(48, 20);
            this.ReplicaSampleDivider.TabIndex = 33;
            this.ReplicaSampleDivider.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.ReplicaSampleDivider.Validating += new System.ComponentModel.CancelEventHandler(this.ReplicaSampleDivider_Validate);
            // 
            // label13
            // 
            this.label13.Location = new System.Drawing.Point(9, 106);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(216, 24);
            this.label13.TabIndex = 32;
            this.label13.Text = "Replica Sample Divider";
            this.label13.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // AllowOverlap
            // 
            this.AllowOverlap.AutoSize = true;
            this.AllowOverlap.Location = new System.Drawing.Point(11, 158);
            this.AllowOverlap.Name = "AllowOverlap";
            this.AllowOverlap.Size = new System.Drawing.Size(88, 17);
            this.AllowOverlap.TabIndex = 34;
            this.AllowOverlap.Text = "AllowOverlap";
            this.AllowOverlap.UseVisualStyleBackColor = true;
            // 
            // DeltaZMultiplier
            // 
            this.DeltaZMultiplier.Location = new System.Drawing.Point(272, 155);
            this.DeltaZMultiplier.Name = "DeltaZMultiplier";
            this.DeltaZMultiplier.Size = new System.Drawing.Size(48, 20);
            this.DeltaZMultiplier.TabIndex = 36;
            this.DeltaZMultiplier.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.DeltaZMultiplier.Validating += new System.ComponentModel.CancelEventHandler(this.DeltaZMultiplier_Validate);
            // 
            // label6
            // 
            this.label6.Location = new System.Drawing.Point(161, 153);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(97, 24);
            this.label6.TabIndex = 35;
            this.label6.Text = "DeltaZ Multiplier";
            this.label6.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // label7
            // 
            this.label7.Location = new System.Drawing.Point(9, 352);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(104, 24);
            this.label7.TabIndex = 37;
            this.label7.Text = "Max. Processors";
            this.label7.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // MaxProcessors
            // 
            this.MaxProcessors.Location = new System.Drawing.Point(104, 355);
            this.MaxProcessors.Name = "MaxProcessors";
            this.MaxProcessors.Size = new System.Drawing.Size(48, 20);
            this.MaxProcessors.TabIndex = 38;
            this.MaxProcessors.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.MaxProcessors.Validating += new System.ComponentModel.CancelEventHandler(this.MaxProcessors_Validate);
            // 
            // MaxTrackingTime
            // 
            this.MaxTrackingTime.Location = new System.Drawing.Point(271, 355);
            this.MaxTrackingTime.Name = "MaxTrackingTime";
            this.MaxTrackingTime.Size = new System.Drawing.Size(48, 20);
            this.MaxTrackingTime.TabIndex = 40;
            this.MaxTrackingTime.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.MaxTrackingTime.Validating += new System.ComponentModel.CancelEventHandler(this.MaxTrackingTime_Validate);
            // 
            // label14
            // 
            this.label14.Location = new System.Drawing.Point(176, 352);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(89, 24);
            this.label14.TabIndex = 39;
            this.label14.Text = "Time Limit (ms)";
            this.label14.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // InitialMultiplicity
            // 
            this.InitialMultiplicity.Location = new System.Drawing.Point(272, 131);
            this.InitialMultiplicity.Name = "InitialMultiplicity";
            this.InitialMultiplicity.Size = new System.Drawing.Size(48, 20);
            this.InitialMultiplicity.TabIndex = 42;
            this.InitialMultiplicity.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
            this.InitialMultiplicity.Validating += new System.ComponentModel.CancelEventHandler(this.InitialMultiplicity_Validate);
            // 
            // label15
            // 
            this.label15.Location = new System.Drawing.Point(161, 128);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(104, 24);
            this.label15.TabIndex = 41;
            this.label15.Text = "Grain Multiplicity";
            this.label15.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // EditConfigForm
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(658, 416);
            this.Controls.Add(this.InitialMultiplicity);
            this.Controls.Add(this.label15);
            this.Controls.Add(this.MaxTrackingTime);
            this.Controls.Add(this.label14);
            this.Controls.Add(this.MaxProcessors);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.DeltaZMultiplier);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.AllowOverlap);
            this.Controls.Add(this.ReplicaSampleDivider);
            this.Controls.Add(this.label13);
            this.Controls.Add(this.MinReplicas);
            this.Controls.Add(this.label12);
            this.Controls.Add(this.ReplicaRadius);
            this.Controls.Add(this.label11);
            this.Controls.Add(this.CellNumY);
            this.Controls.Add(this.CellNumX);
            this.Controls.Add(this.label10);
            this.Controls.Add(this.MyCancelButton);
            this.Controls.Add(this.OKButton);
            this.Controls.Add(this.DelTriggerButton);
            this.Controls.Add(this.NewTriggerButton);
            this.Controls.Add(this.TriggerList);
            this.Controls.Add(this.GrainsDisplay);
            this.Controls.Add(this.MaxArea);
            this.Controls.Add(this.label8);
            this.Controls.Add(this.MinArea);
            this.Controls.Add(this.label9);
            this.Controls.Add(this.MinGrainsHorizontal);
            this.Controls.Add(this.MinGrains01);
            this.Controls.Add(this.MinGrains0);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.CellOverflow);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.XYAlignmentTolerance);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.MaxSlope);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.MinSlope);
            this.Controls.Add(this.label1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "EditConfigForm";
            this.Text = "Edit SmartTracker Configuration";
            this.ResumeLayout(false);
            this.PerformLayout();

		}
		#endregion

		private void CurvePaint(object sender, System.Windows.Forms.PaintEventArgs e)
		{
			int i;
			Graphics g = e.Graphics;
			g.Clear(Color.White);
			if (C == null) return;
			Pen linep = new Pen(Color.CadetBlue, 3);
			Pen axisp = new Pen(Color.DarkGray, 1);
			Pen gridp = new Pen(Color.LightCoral,1);
			Brush textb = new SolidBrush(Color.Black);
			Font numfont = new Font("Arial", 8);
			SizeF ytextsize = new SizeF(0.0f, 0.0f);
			SizeF xtextsize;
			SizeF newsize;
			for (i = 1; i <= Math.Max(C.MinGrainsForHorizontalTrack, C.MinGrainsForVerticalTrack); i++)
			{
				newsize = g.MeasureString(i.ToString(), numfont);
				if (newsize.Width > ytextsize.Width) ytextsize.Width = newsize.Width;
				if (newsize.Height > ytextsize.Height) ytextsize.Height = newsize.Height;
			}
			xtextsize = g.MeasureString("0.1", numfont);
			int ybase = (GrainsDisplay.Height - 12 - (int)xtextsize.Height);
			int xbase = (int)ytextsize.Width + 12;
			double xscale = (double)(GrainsDisplay.Width - xbase - 4 - xtextsize.Width) / C.MaxSlope;
			double yscale = - (double)(ybase - 4) / (double)Math.Max(C.MinGrainsForHorizontalTrack, C.MinGrainsForVerticalTrack);
			Point [] pts = new Point[(int)Math.Ceiling(1 + C.MaxSlope / 0.01)];
			double tsl = 10.0f * (C.MinGrainsForVerticalTrack - C.MinGrainsSlope01) / (C.MinGrainsSlope01 - C.MinGrainsForHorizontalTrack);
			double tinf = C.MinGrainsForHorizontalTrack;
			double td = C.MinGrainsForVerticalTrack - C.MinGrainsForHorizontalTrack;
			for (i = 1; i <= Math.Max(C.MinGrainsForHorizontalTrack, C.MinGrainsForVerticalTrack); i++)
			{
				g.DrawLine(gridp, xbase, ybase + (int)(yscale * i), GrainsDisplay.Width - 4, ybase + (int)(yscale * i));
				g.DrawString(i.ToString(), numfont, textb, 4, ybase + (int)(yscale * i) - (int)(ytextsize.Height) / 2);
			}
			for (i = 1; i * 0.1 <= C.MaxSlope; i++)
			{
				g.DrawLine(gridp, (int)(xbase + i * 0.1 * xscale), ybase, (int)(xbase + i * 0.1 * xscale), 4);
				g.DrawString((i * 0.1).ToString(), numfont, textb, (int)(xbase + i * 0.1 * xscale - (int)g.MeasureString((i * 0.1).ToString(), numfont).Width / 2), ybase + 4);
			}
			g.DrawLine(axisp, xbase, ybase, GrainsDisplay.Width - 4, ybase);
			g.DrawLine(axisp, xbase, 4, xbase, ybase);
			for (i = 0; i < pts.Length; i++)
			{
				pts[i].X = (int)(xbase + i * C.MaxSlope / (pts.Length - 1) * xscale);
				pts[i].Y = ybase + (int)(yscale * (tinf + td / (1 + tsl * i * C.MaxSlope / (pts.Length - 1))));
			}
			g.DrawCurve(linep, pts);
		}

		private void MinSlope_Validate(object sender, System.ComponentModel.CancelEventArgs e)
		{
			try
			{
				double m = Convert.ToDouble(MinSlope.Text);
				if (m < 0 || m > 0.1 || m >= C.MaxSlope) throw new Exception("MinSlope must be between 0 and 0.1 and less than MaxSlope");
				C.MinSlope = m;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				e.Cancel = true;
				return;
			}
			CurvePaint(this, new PaintEventArgs(GrainsDisplay.CreateGraphics(), GrainsDisplay.ClientRectangle));
		}

		private void MaxSlope_Validate(object sender, System.ComponentModel.CancelEventArgs e)
		{
			try
			{
				double m = Convert.ToDouble(MaxSlope.Text);
				if (m < 0 || m > 10.0 || m <= C.MinSlope) throw new Exception("MaxSlope must be between 0 and 10 and greather than MinSlope");
				C.MaxSlope = m;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				e.Cancel = true;
				return;
			}
			CurvePaint(this, new PaintEventArgs(GrainsDisplay.CreateGraphics(), GrainsDisplay.ClientRectangle));		
		}

		private void MinGrains0_Validate(object sender, System.ComponentModel.CancelEventArgs e)
		{
			try
			{
				double m = Convert.ToDouble(MinGrains0.Text);
				if (m < 4 || m > 40) throw new Exception("MinGrains at 0 Slope must be between 4 and 40");
				C.MinGrainsForVerticalTrack = m;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				e.Cancel = true;
				return;
			}
			CurvePaint(this, new PaintEventArgs(GrainsDisplay.CreateGraphics(), GrainsDisplay.ClientRectangle));		
		}

		private void MinGrains01_Validate(object sender, System.ComponentModel.CancelEventArgs e)
		{
			try
			{
				double m = Convert.ToDouble(MinGrains01.Text);
				if (m < 4 || m > 40) throw new Exception("MinGrains at Slope = 0.1 must be between 4 and 40");
				C.MinGrainsSlope01 = m;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				e.Cancel = true;
				return;
			}
			CurvePaint(this, new PaintEventArgs(GrainsDisplay.CreateGraphics(), GrainsDisplay.ClientRectangle));				
		}

		private void MinGrainsHorizontal_Validate(object sender, System.ComponentModel.CancelEventArgs e)
		{
			try
			{
				double m = Convert.ToDouble(MinGrainsHorizontal.Text);
				if (m < 4 || m > 40) throw new Exception("MinGrains for Horizontal Tracks must be between 4 and 40");
				C.MinGrainsForHorizontalTrack = m;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				e.Cancel = true;
				return;
			}
			CurvePaint(this, new PaintEventArgs(GrainsDisplay.CreateGraphics(), GrainsDisplay.ClientRectangle));				
		}

		private void NewTriggerButton_Click(object sender, System.EventArgs e)
		{
			TriggerForm tfrm = new TriggerForm();
			if (tfrm.ShowDialog() == DialogResult.OK)
			{
				Configuration.TriggerInfo [] newtrgs = new Configuration.TriggerInfo[C.Triggers.Length + 1];
				int i;
				for (i = 0; i < C.Triggers.Length; i++)
					newtrgs[i] = C.Triggers[i];
				newtrgs[i].TopLayer = tfrm.TopL;
				newtrgs[i].BottomLayer = tfrm.BottomL;
				newtrgs[i].TriggerLayers = tfrm.TriggersL;
				C.Triggers = newtrgs;
				TriggerList.Items.Add(newtrgs[i].TopLayer.ToString());
				TriggerList.Items[TriggerList.Items.Count - 1].SubItems.Add(newtrgs[i].BottomLayer.ToString());
				string trgtext = "";
				foreach (uint tl in newtrgs[i].TriggerLayers)
				{
					if (trgtext.Length > 0)
						trgtext += ",";
					trgtext += tl.ToString();
				}
				TriggerList.Items[TriggerList.Items.Count - 1].SubItems.Add(trgtext);
			}
		}

		private void DelTriggerButton_Click(object sender, System.EventArgs e)
		{
			if (TriggerList.SelectedItems.Count != 1) return;
			Configuration.TriggerInfo [] newtrgs = new Configuration.TriggerInfo[C.Triggers.Length - 1];
			int i;
			for (i = 0; i < TriggerList.SelectedItems[0].Index; i++)
				newtrgs[i] = C.Triggers[i];
			for (i++; i < C.Triggers.Length; i++)
				newtrgs[i - 1] = C.Triggers[i];
			C.Triggers = newtrgs;
			TriggerList.Items.RemoveAt(TriggerList.SelectedItems[0].Index);

		}

		private void MinArea_Validate(object sender, System.ComponentModel.CancelEventArgs e)
		{
			try
			{
				uint m = Convert.ToUInt32(MinArea.Text);
				if (m < 1 || m > 36 || m > C.MaxArea) throw new Exception("MinArea must be between 1 and 36 and less than MaxArea");
				C.MinArea = m;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				e.Cancel = true;
				return;
			}		
		}

		private void MaxArea_Validate(object sender, System.ComponentModel.CancelEventArgs e)
		{
			try
			{
				uint m = Convert.ToUInt32(MaxArea.Text);
				if (m < 1 || m > 100 || m < C.MinArea) throw new Exception("MaxArea must be between 1 and 100 and greater than MinArea");
				C.MaxArea = m;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				e.Cancel = true;
				return;
			}		
		}

		private void XYAlignmentTolerance_Validate(object sender, System.ComponentModel.CancelEventArgs e)
		{
			try
			{
				double m = Convert.ToDouble(XYAlignmentTolerance.Text);
				if (m < 0.05f || m > 3.0f) throw new Exception("XYAlignmentTolerance must be between 0.05 and 3 micron");
				C.AlignTol = m;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				e.Cancel = true;
				return;
			}		
		}

		private void CellOverflow_Validate(object sender, System.ComponentModel.CancelEventArgs e)
		{
			try
			{
				uint m = Convert.ToUInt32(CellOverflow.Text);
				if (m < 8 || m > 4096) throw new Exception("CellOverflow must be between 8 and 4096");
				C.CellOverflow = m;
			}
			catch (Exception x)
			{
				MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				e.Cancel = true;
				return;
			}				
		}

        private void CellNumX_Validate(object sender, CancelEventArgs e)
        {
            try
            {
                uint m = Convert.ToUInt32(CellNumX.Text);
                if (m < 1 || m > 4096) throw new Exception("CellNumX must be between 1 and 4096");
                C.CellNumX = m;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                e.Cancel = true;
                return;
            }
        }

        private void CellNumY_Validate(object sender, CancelEventArgs e)
        {
            try
            {
                uint m = Convert.ToUInt32(CellNumY.Text);
                if (m < 1 || m > 4096) throw new Exception("CellNumY must be between 1 and 4096");
                C.CellNumY = m;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                e.Cancel = true;
                return;
            }
        }

        private void MinReplicas_Validate(object sender, CancelEventArgs e)
        {
            try
            {
                uint m = Convert.ToUInt32(MinReplicas.Text);
                if (m < 3) throw new Exception("CellNumY must be at least 3");
                C.MinReplicas = m;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                e.Cancel = true;
                return;
            }
        }

        private void ReplicaRadius_Validate(object sender, CancelEventArgs e)
        {
            try
            {
                double m = Convert.ToDouble(ReplicaRadius.Text);
                if (m < 0.0 || m > 3.0) throw new Exception("ReplicaRadius must be between 0.0 and 3.0 micron");
                C.ReplicaRadius = m;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                e.Cancel = true;
                return;
            }		
        }

        private void ReplicaSampleDivider_Validate(object sender, CancelEventArgs e)
        {
            try
            {
                uint m = Convert.ToUInt32(ReplicaSampleDivider.Text);
                if (m < 1) throw new Exception("ReplicaSampleDivider must be at least 1");
                C.ReplicaSampleDivider = m;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                e.Cancel = true;
                return;
            }
        }

        private void DeltaZMultiplier_Validate(object sender, CancelEventArgs e)
        {
            try
            {
                double m = Convert.ToDouble(DeltaZMultiplier.Text);
                if (m < 0.5 || m > 2.0) throw new Exception("DeltaZMultiplier must be between 0.5 and 2.0 micron");
                C.DeltaZMultiplier = m;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                e.Cancel = true;
                return;
            }		
        }

        private void MaxProcessors_Validate(object sender, CancelEventArgs e)
        {
            try
            {
                uint m = Convert.ToUInt32(MaxProcessors.Text);
                if (m < 1) throw new Exception("MaxProcessors must be 0 for automatic assigment of processors, or a positive number.");
                C.MaxProcessors = m;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                e.Cancel = true;
                return;
            }
        }

        private void MaxTrackingTime_Validate(object sender, CancelEventArgs e)
        {
            try
            {
                uint m = Convert.ToUInt32(MaxTrackingTime.Text);
                if (m < 100) throw new Exception("Time Limit must be at least 100 ms.");
                C.MaxTrackingTimeMS = (int)m;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                e.Cancel = true;
                return;
            }
        }

        private void InitialMultiplicity_Validate(object sender, CancelEventArgs e)
        {
            try
            {
                uint m = Convert.ToUInt32(InitialMultiplicity.Text);
                if (m < 1) throw new Exception("Grain Multiplicity must be at least 1.");
                C.InitialMultiplicity = m;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                e.Cancel = true;
                return;
            }
        }

        private void MyCancelButton_Click(object sender, System.EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        private void OKButton_Click(object sender, System.EventArgs e)
        {
            C.AllowOverlap = AllowOverlap.Checked;
            DialogResult = DialogResult.OK;
            Close();
        }
	}
}
