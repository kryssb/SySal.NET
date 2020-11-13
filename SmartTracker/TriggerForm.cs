using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.Processing.SmartTracking
{
	/// <summary>
	/// Summary description for TriggerForm.
	/// </summary>
	public class TriggerForm : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.TextBox TopLayer;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.TextBox BottomLayer;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.TextBox TriggerLayers;
		private System.Windows.Forms.Button OKButton;

		public uint TopL, BottomL;
		public uint [] TriggersL;
		private System.Windows.Forms.Button MyCancelButton;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public TriggerForm()
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
			this.TopLayer = new System.Windows.Forms.TextBox();
			this.label2 = new System.Windows.Forms.Label();
			this.BottomLayer = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.TriggerLayers = new System.Windows.Forms.TextBox();
			this.OKButton = new System.Windows.Forms.Button();
			this.MyCancelButton = new System.Windows.Forms.Button();
			this.SuspendLayout();
			// 
			// label1
			// 
			this.label1.Location = new System.Drawing.Point(8, 8);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(88, 16);
			this.label1.TabIndex = 0;
			this.label1.Text = "Top Layer";
			this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// TopLayer
			// 
			this.TopLayer.Location = new System.Drawing.Point(112, 8);
			this.TopLayer.Name = "TopLayer";
			this.TopLayer.Size = new System.Drawing.Size(56, 20);
			this.TopLayer.TabIndex = 1;
			this.TopLayer.Text = "2";
			this.TopLayer.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label2
			// 
			this.label2.Location = new System.Drawing.Point(8, 40);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(88, 16);
			this.label2.TabIndex = 2;
			this.label2.Text = "Bottom Layer";
			this.label2.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// BottomLayer
			// 
			this.BottomLayer.Location = new System.Drawing.Point(112, 40);
			this.BottomLayer.Name = "BottomLayer";
			this.BottomLayer.Size = new System.Drawing.Size(56, 20);
			this.BottomLayer.TabIndex = 3;
			this.BottomLayer.Text = "12";
			this.BottomLayer.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// label3
			// 
			this.label3.Location = new System.Drawing.Point(8, 72);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(200, 16);
			this.label3.TabIndex = 4;
			this.label3.Text = "Trigger Layers (comma separated)";
			this.label3.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
			// 
			// TriggerLayers
			// 
			this.TriggerLayers.Location = new System.Drawing.Point(8, 96);
			this.TriggerLayers.Name = "TriggerLayers";
			this.TriggerLayers.Size = new System.Drawing.Size(200, 20);
			this.TriggerLayers.TabIndex = 5;
			this.TriggerLayers.Text = "5,9";
			this.TriggerLayers.TextAlign = System.Windows.Forms.HorizontalAlignment.Right;
			// 
			// OKButton
			// 
			this.OKButton.Location = new System.Drawing.Point(184, 8);
			this.OKButton.Name = "OKButton";
			this.OKButton.Size = new System.Drawing.Size(96, 24);
			this.OKButton.TabIndex = 6;
			this.OKButton.Text = "OK";
			this.OKButton.Click += new System.EventHandler(this.OKButton_Click);
			// 
			// MyCancelButton
			// 
			this.MyCancelButton.Location = new System.Drawing.Point(184, 40);
			this.MyCancelButton.Name = "MyCancelButton";
			this.MyCancelButton.Size = new System.Drawing.Size(96, 24);
			this.MyCancelButton.TabIndex = 7;
			this.MyCancelButton.Text = "Cancel";
			this.MyCancelButton.Click += new System.EventHandler(this.CancelButton_Click);
			// 
			// TriggerForm
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(296, 126);
			this.Controls.AddRange(new System.Windows.Forms.Control[] {
																		  this.MyCancelButton,
																		  this.OKButton,
																		  this.TriggerLayers,
																		  this.label3,
																		  this.BottomLayer,
																		  this.label2,
																		  this.TopLayer,
																		  this.label1});
			this.Name = "TriggerForm";
			this.Text = "New Trigger";
			this.ResumeLayout(false);

		}
		#endregion

		private void OKButton_Click(object sender, System.EventArgs e)
		{
			try
			{
				TopL = Convert.ToUInt32(TopLayer.Text);
				BottomL = Convert.ToUInt32(BottomLayer.Text);
				string [] trg = TriggerLayers.Text.Split(',');
				TriggersL = new uint[trg.Length];
				int i;
				for (i = 0; i < trg.Length; i++)
					TriggersL[i] = Convert.ToUInt32(trg[i]);
				DialogResult = DialogResult.OK;
				Close();
			}
			catch (Exception)
			{
				MessageBox.Show("Invalid input", "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
				return;
			}
		}

		private void CancelButton_Click(object sender, System.EventArgs e)
		{
			DialogResult = DialogResult.Cancel;
			Close();
		}
	}
}
