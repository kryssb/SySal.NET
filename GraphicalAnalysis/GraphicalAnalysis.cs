using System;
using System.Drawing;

namespace NumericalTools
{
	/// <summary>
	/// A library for producing commonly used plots.
	/// </summary>
	public class Plot
	{
		private Pen myPen = new Pen(System.Drawing.Color.Black, 1);
		private Pen myRedPen = new Pen(System.Drawing.Color.Red, 1);

        private Marker m_Marker = new Marker.FilledCircle();

        public string CurrentMarker
        {
            get { return m_Marker.MarkerType; }
            set
            {
                m_Marker = Marker.GetMarker(value);
            }
        }

        private uint m_MarkerSize = 3;

        public uint CurrentMarkerSize
        {
            get { return m_MarkerSize; }
            set
            {
                m_MarkerSize = value;
            }
        }

        private bool m_ShowErrorBars = true;

        public bool ShowErrorBars
        {
            get { return m_ShowErrorBars; }
            set
            {
                m_ShowErrorBars = value;
            }
        }


		public enum PaletteType { RGBContinuous = 0, Flat16 = 1, GreyContinuous = 2, Grey16 = 3 }

		private PaletteType m_Palette = PaletteType.RGBContinuous;

		public PaletteType Palette
		{
			get { return m_Palette; }
			set 
			{
				switch (value)
				{
					case PaletteType.RGBContinuous: HueIndexTable = Plot.InitRGBContinuousHueIndexTable(); break;
					case PaletteType.Flat16: HueIndexTable = Plot.InitFlatHueIndexTable(); break;
					case PaletteType.GreyContinuous: HueIndexTable = Plot.InitGreyContinuousIndexTable(); break;
					case PaletteType.Grey16: HueIndexTable = Plot.InitGreyFlatIndexTable(); break;
					default: throw new Exception("Unknown palette");
				}
				m_Palette = value;
			}
		}
		
		private Font m_LabelFont = new Font("Comic Sans MS", 9, System.Drawing.FontStyle.Bold);

		public Font LabelFont
		{
			get { return (m_LabelFont == null) ? null : (Font)m_LabelFont.Clone(); }
			set 
			{ 
				m_LabelFont = value; 
				if (m_LabelFont == null)
					m_LabelFont = new Font("Comic Sans MS", 9, System.Drawing.FontStyle.Bold);
			}
		}

		private Brush myBrush = new SolidBrush(System.Drawing.Color.Black);
		private Brush myRedBrush = new SolidBrush(System.Drawing.Color.Red);

		public Plot()
		{
			//
			// TODO: Add constructor logic here
			//
		}

		#region Component Methods


		public double [][] Histo(System.Drawing.Graphics gPB, int Width, int Height)
		{
            SetQuality(gPB);
			double [] X_Mean, tmpX_Mean, Y_Vec, tmpY_Vec, N_Y_Vec, tmpN_Y_Vec;
			double MinX =0, MaxY =0;
			double MedX =0, RMSx =0;
			double MaxX =0, MinY =0;
			double dum =0;
			int i=0;
			double[] Gau= new double[2];
			int cc=0,j;
			
			//gPB.Clear(Color.White);

			//Prepara L'Istogramma
			Fitting.Prepare_Custom_Distribution(m_VecX, 1, m_DX, 0, out tmpX_Mean, out tmpY_Vec, out tmpN_Y_Vec);
			Fitting.FindStatistics(m_VecX, ref MaxX, ref MinX, ref MedX, ref RMSx);
			MinX=MinX-m_DX/2; 
			MaxX=MaxX+m_DX/2; 
			//Se quelli predisposti dall'esterno 
			//non sono buoni allora usa quelli dall'interno.
			//viceversa vanno bene quelli di default calcolati or ora

			/* HERE!!! arrotondamento di SpazioLabelX a potenze di 2, 5 o 10
			 * double ArSpazioLabelX = Math.Pow(10.0, Math.Floor(Math.Log10(SpazioLabelX)));
			 * if (ArSpazioLabelX * 5 < SpazioLabelX) SpazioLabelX = 5 * ArSpazioLabelX;
			 * else if (ArSpazioLabelX * 2 < SpazioLabelX) SpazioLabelX = 2 * ArSpazioLabelX;
			 * else SpazioLabelX = ArSpazioLabelX;
			 *
			 * MinX = Math.Floor(MinX / SpazioLabelX) * SpazioLabelX;
			 * fine arrotondamento
			 */


			if ((m_MaxX-m_MinX)>=m_DX)
			{
				MinX=m_MinX-m_DX/2; 
				MaxX=m_MaxX+m_DX/2;
		
				for(j = 0; j<tmpX_Mean.GetLength(0); j++) if (tmpX_Mean[j]>MinX && tmpX_Mean[j]<MaxX ) cc++;

				Y_Vec= new double[cc];
				X_Mean= new double[cc];
				N_Y_Vec= new double[cc];
				cc=0;
				for(j = 0; j<tmpX_Mean.GetLength(0); j++)
					if (tmpX_Mean[j]>MinX && tmpX_Mean[j]<MaxX )
					{
						X_Mean[cc]=tmpX_Mean[j];
						Y_Vec[cc]=tmpY_Vec[j];
						N_Y_Vec[cc]=tmpN_Y_Vec[j];
						cc++;
					};

			}
			else
			{
				j=tmpX_Mean.GetLength(0);
				N_Y_Vec= new double[j];
				Y_Vec= new double[j];
				X_Mean= new double[j];
				N_Y_Vec= (double[])tmpN_Y_Vec.Clone();
				Y_Vec= (double[])tmpY_Vec.Clone();
				X_Mean= (double[])tmpX_Mean.Clone();
			};

			Fitting.FindStatistics(Y_Vec, ref MaxY, ref MinY, ref dum, ref dum);

			if (m_HistoFit== 0)
			{
				m_FitPar= new double [4];
				m_ParDescr= new string [4];
			}
			else if(m_HistoFit==-1)
			{
				m_FitPar= new double [5];
				m_ParDescr= new string [5];
			}
			else if(m_HistoFit==-2)
			{
				m_FitPar= new double [2];
				m_ParDescr= new string [2];
			};
			m_ParDescr[0]="Mean Value";
			m_ParDescr[1]="RMS";
			m_FitPar[0]=MedX;
			m_FitPar[1]=RMSx;

			//passa i parametri generali a quelli locali
			/*
			*Prepara i limiti verticali ed orizzontali'
			*/

			float AggX=0, AggY=0, LengthX=0, LengthY=0;
			float AggFontX=0, AggFontY=0;
			string StrY="";

			SetBorders(gPB, MaxX, MinX, MaxY, 0, Height, Width, 
				ref AggX, ref AggY, ref LengthX, ref LengthY,
				ref AggFontX, ref AggFontY, ref StrY);

			int n=X_Mean.GetLength(0);

            double[][] o_vals = new double[3][] { new double[n], new double[n], new double[n] };

			//labels sugli assi
			System.Drawing.PointF myPt = new System.Drawing.PointF();
			myPt.X = 0;
			myPt.Y = 0;
			gPB.DrawString("Counts", m_LabelFont, myBrush, myPt);

			myPt.X = Width - gPB.MeasureString(m_XTitle,m_LabelFont).Width;
			myPt.Y = Height -gPB.MeasureString(m_XTitle,m_LabelFont).Height;
			gPB.DrawString(m_XTitle, m_LabelFont, myBrush, myPt);

			//il minimo deve sempre essere zero MinY=0
			DrawXAxis(n, 0, MaxX, MinX, MaxY, 0, AggX, AggY, AggFontX, AggFontY, gPB);
			//Aggiunto dopo
			StrY="F0";
			DrawYAxis(MinX, /*MaxX, MinX,*/ MaxY, 0, AggX, AggY, AggFontX, AggFontY, StrY, gPB);

			// Disegna l'Istogramma
            double cum = 0.0;
            for (i = 0; i < X_Mean.GetLength(0); i++)
            {
                o_vals[0][i] = X_Mean[i];
                o_vals[1][i] = Y_Vec[i];
                o_vals[2][i] = (cum += Y_Vec[i]);
                if (((X_Mean[i] - m_DX / 2) >= MinX) && ((X_Mean[i] + m_DX / 2) <= MaxX))
                    if (m_HistoFill == false)
                        gPB.DrawRectangle(myRedPen, AffineX(X_Mean[i] - m_DX / 2), AffineY(Y_Vec[i]), ShrinkX(m_DX), ShrinkY(Y_Vec[i]));
                    else
                        gPB.FillRectangle(myRedBrush, AffineX(X_Mean[i] - m_DX / 2), AffineY(Y_Vec[i]), ShrinkX(m_DX), ShrinkY(Y_Vec[i]));
            }

			// Disegna il Fit
			if (m_HistoFit> -2)
				if (m_HistoFit== 0)
					Gauss_Fit_Histo(MaxX, MinX, X_Mean, Y_Vec, N_Y_Vec, gPB);
				else if (m_HistoFit== -1)
					InvGauss_Fit_Histo(MaxX, MinX, X_Mean, Y_Vec, N_Y_Vec, gPB);
				else
				{
					int[] Ent= new int[n];
					double[] DSY= new double[n];
					for( i = 0; i< n;i++)
					{
						Ent[i]=1;
						DSY[i]=1;
					};
					Fit_Plot_Scatter(X_Mean, Y_Vec, Ent, DSY, MaxX, MinX, MaxY, MinY, m_HistoFit, gPB);
				};

			// Disegna la funzione al di sopra dell'isto
			if (m_FunctionOverlayed) OverlayFunction(MaxX, MinX, MaxY, MinY, gPB);

			//All'esterno: per i click sul plot
			m_PlottedX = (double[])X_Mean.Clone();
			m_PlottedY = (double[])Y_Vec.Clone();
			
			DrawPanel(gPB, 1);

            return o_vals;
		}

        public double [][] HistoSkyline(System.Drawing.Graphics gPB, int Width, int Height)
        {
            SetQuality(gPB);
            double[] X_Mean, tmpX_Mean, Y_Vec, tmpY_Vec, N_Y_Vec, tmpN_Y_Vec;
            double MinX = 0, MaxY = 0;
            double MedX = 0, RMSx = 0;
            double MaxX = 0, MinY = 0;
            double dum = 0;
            int i = 0;
            double[] Gau = new double[2];
            int cc = 0, j;

            //gPB.Clear(Color.White);

            //Prepara L'Istogramma
            Fitting.Prepare_Custom_Distribution(m_VecX, 1, m_DX, 0, out tmpX_Mean, out tmpY_Vec, out tmpN_Y_Vec);
            Fitting.FindStatistics(m_VecX, ref MaxX, ref MinX, ref MedX, ref RMSx);
            MinX = MinX - m_DX / 2;
            MaxX = MaxX + m_DX / 2;
            //Se quelli predisposti dall'esterno 
            //non sono buoni allora usa quelli dall'interno.
            //viceversa vanno bene quelli di default calcolati or ora

            /* HERE!!! arrotondamento di SpazioLabelX a potenze di 2, 5 o 10
             * double ArSpazioLabelX = Math.Pow(10.0, Math.Floor(Math.Log10(SpazioLabelX)));
             * if (ArSpazioLabelX * 5 < SpazioLabelX) SpazioLabelX = 5 * ArSpazioLabelX;
             * else if (ArSpazioLabelX * 2 < SpazioLabelX) SpazioLabelX = 2 * ArSpazioLabelX;
             * else SpazioLabelX = ArSpazioLabelX;
             *
             * MinX = Math.Floor(MinX / SpazioLabelX) * SpazioLabelX;
             * fine arrotondamento
             */


            if ((m_MaxX - m_MinX) >= m_DX)
            {
                MinX = m_MinX - m_DX / 2;
                MaxX = m_MaxX + m_DX / 2;

                for (j = 0; j < tmpX_Mean.GetLength(0); j++) if (tmpX_Mean[j] > MinX && tmpX_Mean[j] < MaxX) cc++;

                Y_Vec = new double[cc];
                X_Mean = new double[cc];
                N_Y_Vec = new double[cc];
                cc = 0;
                for (j = 0; j < tmpX_Mean.GetLength(0); j++)
                    if (tmpX_Mean[j] > MinX && tmpX_Mean[j] < MaxX)
                    {
                        X_Mean[cc] = tmpX_Mean[j];
                        Y_Vec[cc] = tmpY_Vec[j];
                        N_Y_Vec[cc] = tmpN_Y_Vec[j];
                        cc++;
                    };

            }
            else
            {
                j = tmpX_Mean.GetLength(0);
                N_Y_Vec = new double[j];
                Y_Vec = new double[j];
                X_Mean = new double[j];
                N_Y_Vec = (double[])tmpN_Y_Vec.Clone();
                Y_Vec = (double[])tmpY_Vec.Clone();
                X_Mean = (double[])tmpX_Mean.Clone();
            };

            Fitting.FindStatistics(Y_Vec, ref MaxY, ref MinY, ref dum, ref dum);

            if (m_HistoFit == 0)
            {
                m_FitPar = new double[4];
                m_ParDescr = new string[4];
            }
            else if (m_HistoFit == -1)
            {
                m_FitPar = new double[5];
                m_ParDescr = new string[5];
            }
            else if (m_HistoFit == -2)
            {
                m_FitPar = new double[2];
                m_ParDescr = new string[2];
            };
            m_ParDescr[0] = "Mean Value";
            m_ParDescr[1] = "RMS";
            m_FitPar[0] = MedX;
            m_FitPar[1] = RMSx;

            //passa i parametri generali a quelli locali
            /*
            *Prepara i limiti verticali ed orizzontali'
            */

            float AggX = 0, AggY = 0, LengthX = 0, LengthY = 0;
            float AggFontX = 0, AggFontY = 0;
            string StrY = "";

            SetBorders(gPB, MaxX, MinX, MaxY, 0, Height, Width,
                ref AggX, ref AggY, ref LengthX, ref LengthY,
                ref AggFontX, ref AggFontY, ref StrY);

            int n = X_Mean.GetLength(0);

            double[][] o_vals = new double[3][] { new double[n], new double[n], new double[n] };

            //labels sugli assi
            System.Drawing.PointF myPt = new System.Drawing.PointF();
            myPt.X = 0;
            myPt.Y = 0;
            gPB.DrawString("Counts", m_LabelFont, myBrush, myPt);

            myPt.X = Width - gPB.MeasureString(m_XTitle, m_LabelFont).Width;
            myPt.Y = Height - gPB.MeasureString(m_XTitle, m_LabelFont).Height;
            gPB.DrawString(m_XTitle, m_LabelFont, myBrush, myPt);

            //il minimo deve sempre essere zero MinY=0
            DrawXAxis(n, 0, MaxX, MinX, MaxY, 0, AggX, AggY, AggFontX, AggFontY, gPB);
            //Aggiunto dopo
            StrY = "F0";
            DrawYAxis(MinX, /*MaxX, MinX,*/ MaxY, 0, AggX, AggY, AggFontX, AggFontY, StrY, gPB);

            // Disegna l'Istogramma
            double cum = 0.0;
            for (i = 0; i < X_Mean.GetLength(0); i++)
            {
                o_vals[0][i] = X_Mean[i];
                o_vals[1][i] = Y_Vec[i];
                o_vals[2][i] = (cum += Y_Vec[i]);
                if (((X_Mean[i] - m_DX / 2) >= MinX) && ((X_Mean[i] + m_DX / 2) <= MaxX))
                {
                    if (i > 0) gPB.DrawLine(myRedPen, AffineX(X_Mean[i] - m_DX / 2), AffineY(Y_Vec[i - 1]), AffineX(X_Mean[i] - m_DX / 2), AffineY(Y_Vec[i]));
                    gPB.DrawLine(myRedPen, AffineX(X_Mean[i] - m_DX / 2), AffineY(Y_Vec[i]), AffineX(X_Mean[i] + m_DX / 2), AffineY(Y_Vec[i]));
                }
            }

            // Disegna il Fit
            if (m_HistoFit > -2)
                if (m_HistoFit == 0)
                    Gauss_Fit_Histo(MaxX, MinX, X_Mean, Y_Vec, N_Y_Vec, gPB);
                else if (m_HistoFit == -1)
                    InvGauss_Fit_Histo(MaxX, MinX, X_Mean, Y_Vec, N_Y_Vec, gPB);
                else
                {
                    int[] Ent = new int[n];
                    double[] DSY = new double[n];
                    for (i = 0; i < n; i++)
                    {
                        Ent[i] = 1;
                        DSY[i] = 1;
                    };
                    Fit_Plot_Scatter(X_Mean, Y_Vec, Ent, DSY, MaxX, MinX, MaxY, MinY, m_HistoFit, gPB);
                };

            // Disegna la funzione al di sopra dell'isto
            if (m_FunctionOverlayed) OverlayFunction(MaxX, MinX, MaxY, MinY, gPB);

            //All'esterno: per i click sul plot
            m_PlottedX = (double[])X_Mean.Clone();
            m_PlottedY = (double[])Y_Vec.Clone();

            DrawPanel(gPB, 1);

            return o_vals;
        }


		public double [][] GroupScatter(System.Drawing.Graphics gPB, int Width, int Height)
		{
            SetQuality(gPB);
			double [] X_Mean, Y_Mean, SY_Vec;
			int [] Ent;
			double MinX =0, MaxY =0,MaxSY =0, MedX =0, RMSx =0;
			double MaxX =0, MinY =0, MedY =0, RMSy =0, dum=0;
			int i=0;
			//double[] Pars = new double[3];
			//int[] Iter= new int[2];
			//double[] Gau= new double[2];
			//Nuovo Codice


			//gPB.Clear(Color.White);

			Fitting.GroupScatter(m_VecX, m_VecY,1, m_DX, 0, 1, 
				out X_Mean, out Y_Mean, 
				out SY_Vec, out Ent);
			if (m_SetXDefaultLimits || (!m_SetXDefaultLimits && m_MaxX<=m_MinX) )
				Fitting.FindStatistics(m_VecX, ref m_MaxX, ref m_MinX, ref MedX, ref RMSx);
			if (m_SetYDefaultLimits || (!m_SetYDefaultLimits && m_MaxY<=m_MinY) )
				Fitting.FindStatistics(m_VecY, ref m_MaxY, ref m_MinY, ref MedY, ref RMSy);
			Fitting.FindStatistics(SY_Vec, ref MaxSY, ref dum, ref dum, ref dum);


			//Pulisce i vecchi parametri
			m_FitPar = new double [1];
			m_ParDescr = new string [1];

			//Aggiunge un po' di spazio ai bordi
			//per non far disegnare i dati fino
			//alle estremita': il 10% del range
			MinX=m_MinX; MaxX=m_MaxX; 
			MinY=m_MinY; MaxY=m_MaxY;

			MinX = MinX -m_DX/2;
			MaxX = MaxX +m_DX/2;

			//Chiamata
			float AggX=0, AggY=0, LengthX=0, LengthY=0;
			float AggFontX=0, AggFontY=0;
			string StrY="";

			SetBorders(gPB, MaxX, MinX, MaxY, MinY, Height, Width, 
				ref AggX, ref AggY, ref LengthX, ref LengthY,
				ref AggFontX, ref AggFontY, ref StrY);

			System.Drawing.PointF myPt = new System.Drawing.PointF();
			myPt.X = 0;
			myPt.Y = 0;
			gPB.DrawString(m_YTitle, m_LabelFont, myBrush, myPt);

			myPt.X = Width - gPB.MeasureString(m_XTitle,m_LabelFont).Width;
			myPt.Y = Height -gPB.MeasureString(m_XTitle,m_LabelFont).Height;
			gPB.DrawString(m_XTitle, m_LabelFont, myBrush, myPt);
			DrawXAxis((int)((MaxX-MinX)/m_DX), MinY, MaxX, MinX, MaxY, MinY, AggX, AggY, AggFontX, AggFontY, gPB);
			DrawYAxis(MinX, /*MaxX, MinX,*/ MaxY, MinY, AggX, AggY, AggFontX, AggFontY, StrY, gPB);

			int n=X_Mean.GetLength(0);

            double[][] o_vals = new double[4][] { new double[n], new double[n], new double[n], new double[n] };

            for (i = 0; i < n; i++)
            {
                o_vals[0][i] = X_Mean[i];
                o_vals[1][i] = Y_Mean[i];
                o_vals[2][i] = SY_Vec[i];
                o_vals[3][i] = Ent[i];
                if (Ent[i] > 0 && ((X_Mean[i] - m_DX / 2) >= MinX) && ((X_Mean[i] + m_DX / 2) <= MaxX))
                {
                    if (Ent[i] == 1 || Ent[i] == 2) SY_Vec[i] = MaxSY;
                    //Ricorda il numero dei bin non vuoti 

                    float s = (Ent[i] > 1) ? ShrinkY(SY_Vec[i] / Math.Sqrt(Ent[i] - 1)) : 10000.0f;

                    if (m_ShowErrorBars)
                    {
                        gPB.DrawLine(myPen, AffineX(X_Mean[i]), AffineY(Y_Mean[i]) + s, AffineX(X_Mean[i]), AffineY(Y_Mean[i]) - s);
                        gPB.DrawLine(myPen, AffineX(X_Mean[i] - m_DX / 4), AffineY(Y_Mean[i]) + s, AffineX(X_Mean[i] + m_DX / 4), AffineY(Y_Mean[i]) + s);
                        gPB.DrawLine(myPen, AffineX(X_Mean[i] - m_DX / 4), AffineY(Y_Mean[i]) - s, AffineX(X_Mean[i] + m_DX / 4), AffineY(Y_Mean[i]) - s);
                    }
                    //gPB.DrawEllipse(myPen, AffineX(X_Mean[i] - m_DX / 16), AffineY(Y_Mean[i] + m_DY / 16), ShrinkX(m_DX / 8), ShrinkY(m_DY / 8));
                    m_Marker.Draw(gPB, myPen, myBrush, m_MarkerSize, (float)AffineX(X_Mean[i]), (float)AffineY(Y_Mean[i]));

                    /*
                                        gPB.DrawLine(myPen, AffineX(X_Mean[i]),AffineY(Y_Mean[i]+SY_Vec[i]),AffineX(X_Mean[i]),AffineY(Y_Mean[i]-SY_Vec[i]));
                                        gPB.DrawLine(myPen, AffineX(X_Mean[i]-m_DX/4),AffineY(Y_Mean[i]+SY_Vec[i]),AffineX(X_Mean[i]+m_DX/4),AffineY(Y_Mean[i]+SY_Vec[i]));
                                        gPB.DrawLine(myPen, AffineX(X_Mean[i]-m_DX/4),AffineY(Y_Mean[i]-SY_Vec[i]),AffineX(X_Mean[i]+m_DX/4),AffineY(Y_Mean[i]-SY_Vec[i]));
                                        gPB.DrawEllipse(myPen,AffineX(X_Mean[i]-m_DX/8),AffineY(Y_Mean[i]+m_DY/8),ShrinkX(m_DX/4),ShrinkY(m_DY/4));
                    */
                };
            }

			if (m_ScatterFit>0) Fit_Plot_Scatter(X_Mean, Y_Mean, Ent, SY_Vec, 
									MaxX, MinX, MaxY, MinY, m_ScatterFit, gPB);

			if (m_FunctionOverlayed) OverlayFunction(MaxX, MinX, MaxY, MinY, gPB);

			//All'esterno
			m_PlottedX = (double[])X_Mean.Clone();
			m_PlottedY = (double[])Y_Mean.Clone();
			m_PlottedSY = (double[])SY_Vec.Clone();

			DrawPanel(gPB, 2);

            return o_vals;
		}

        private int m_JoinPenThickness = 0;

        public int JoinPenThickness
        {
            get { return m_JoinPenThickness; }
            set
            {
                m_JoinPenThickness = value;
            }
        }

		public double [][] Scatter(System.Drawing.Graphics gPB, int Width, int Height)
		{
            SetQuality(gPB);
			double MinX =0, MaxY =0, MedX =0, RMSx =0;
			double MaxX =0, MinY =0, MedY =0, RMSy =0;
			int i=0;
			//double[] Pars = new double[3];
			//int[] Iter= new int[2];
			//double[] Gau= new double[2];
			//Nuovo Codice

			//gPB.Clear(Color.White);

			if (m_SetXDefaultLimits || (!m_SetXDefaultLimits && m_MaxX<=m_MinX) )
				Fitting.FindStatistics(m_VecX, ref m_MaxX, ref m_MinX, ref MedX, ref RMSx);
			if (m_SetYDefaultLimits || (!m_SetYDefaultLimits && m_MaxY<=m_MinY) )
				Fitting.FindStatistics(m_VecY, ref m_MaxY, ref m_MinY, ref MedY, ref RMSy);

			//Pulisce i vecchi parametri
			m_FitPar = new double [1];
			m_ParDescr = new string [1];

			MinX = m_MinX; MaxX = m_MaxX; 
			MinY = m_MinY; MaxY = m_MaxY;

			float AggX=0, AggY=0, LengthX=0, LengthY=0;
			float AggFontX=0, AggFontY=0;
			string StrY="";

			SetBorders(gPB, MaxX, MinX, MaxY, MinY, Height, Width, 
				ref AggX, ref AggY, ref LengthX, ref LengthY,
				ref AggFontX, ref AggFontY, ref StrY);

			System.Drawing.PointF myPt = new System.Drawing.PointF();
			myPt.X = 0;
			myPt.Y = 0;
			gPB.DrawString(m_YTitle, m_LabelFont, myBrush, myPt);

			myPt.X = Width - gPB.MeasureString(m_XTitle,m_LabelFont).Width;
			myPt.Y = Height -gPB.MeasureString(m_XTitle,m_LabelFont).Height;
			gPB.DrawString(m_XTitle, m_LabelFont, myBrush, myPt);

			DrawXAxis(100, MinY, MaxX, MinX, MaxY, MinY, AggX, AggY, AggFontX, AggFontY, gPB);
			DrawYAxis(MinX, /*MaxX, MinX,*/ MaxY, MinY, AggX, AggY, AggFontX, AggFontY, StrY, gPB);

            Pen jpen = null;
            if (m_JoinPenThickness > 0)
                jpen = new Pen(myPen.Color, m_JoinPenThickness);
			int n = m_VecX.GetLength(0);
			double[] SY_Vec = new double[n];
			int[] Ent = new int[n];
			for(i=0; i<n;i++)
			{
				Ent[i]=1;
				SY_Vec[i]=1;
                if ((m_VecX[i] >= m_MinX) && (m_VecX[i] <= m_MaxX) &&
                    (m_VecY[i] >= m_MinY) && (m_VecY[i] <= m_MaxY))
                {
                    if (jpen == null)
                        /*gPB.FillEllipse(myBrush, AffineX(m_VecX[i]) - ShrinkX((MaxX - MinX) / 200), AffineY(m_VecY[i]) - ShrinkY((MaxY - MinY) / 200), ShrinkX((MaxX - MinX) / 100), ShrinkY((MaxY - MinY) / 100));*/
                        m_Marker.Draw(gPB, myPen, myBrush, m_MarkerSize, (float)AffineX(m_VecX[i]), (float)AffineY(m_VecY[i]));
                    else if (i > 0) gPB.DrawLine(jpen, AffineX(m_VecX[i]), AffineY(m_VecY[i]), AffineX(m_VecX[i - 1]), AffineY(m_VecY[i - 1]));
                    if (m_ShowErrorBars)
                    {
                        gPB.DrawLine(myPen, AffineX(m_VecX[i]), AffineY(m_VecY[i] + m_VecDX[i]), AffineX(m_VecX[i]), AffineY(m_VecY[i] - m_VecDY[i]));
                        gPB.DrawLine(myPen, AffineX(m_VecX[i] - m_DX / 4), AffineY(m_VecY[i] + m_VecDX[i]), AffineX(m_VecX[i] + m_DX / 4), AffineY(m_VecY[i] + m_VecDX[i]));
                        gPB.DrawLine(myPen, AffineX(m_VecX[i] - m_DX / 4), AffineY(m_VecY[i] - m_VecDY[i]), AffineX(m_VecX[i] + m_DX / 4), AffineY(m_VecY[i] - m_VecDY[i]));
                    }
                }
			};


			if (m_ScatterFit>0) Fit_Plot_Scatter(m_VecX, m_VecY, Ent, SY_Vec, 
									MaxX, MinX, MaxY, MinY, m_ScatterFit, gPB);

			if (m_FunctionOverlayed) OverlayFunction(MaxX, MinX, MaxY, MinY, gPB);
			//All'esterno
			m_PlottedX = (double[])m_VecX.Clone();
			m_PlottedY = (double[])m_VecY.Clone();
			m_PlottedSY = (double[])SY_Vec.Clone();

			DrawPanel(gPB, 2);

            return null;
		}

        public double [][] ScatterHue(System.Drawing.Graphics gPB, int Width, int Height)
        {
            SetQuality(gPB);
            double MinX = 0, MaxX = 0, MedX = 0, RMSx = 0;
            double MinY = 0, MaxY = 0, MedY = 0, RMSy = 0;
            double MinZ = 0, MaxZ = 0, MedZ = 0, RMSz = 0;
            int i = 0;
            //double[] Pars = new double[3];
            //int[] Iter= new int[2];
            //double[] Gau= new double[2];
            //Nuovo Codice

            //gPB.Clear(Color.White);

            if (m_SetXDefaultLimits || (!m_SetXDefaultLimits && m_MaxX <= m_MinX))
                Fitting.FindStatistics(m_VecX, ref m_MaxX, ref m_MinX, ref MedX, ref RMSx);
            if (m_SetYDefaultLimits || (!m_SetYDefaultLimits && m_MaxY <= m_MinY))
                Fitting.FindStatistics(m_VecY, ref m_MaxY, ref m_MinY, ref MedY, ref RMSy);            
            Fitting.FindStatistics(m_VecZ, ref MaxZ, ref MinZ, ref MedZ, ref RMSz);
            if (MaxZ <= MinZ) MaxZ = Math.Abs(MinZ) * 2 + 1;

            //Pulisce i vecchi parametri
            m_FitPar = new double[1];
            m_ParDescr = new string[1];

            MinX = m_MinX; MaxX = m_MaxX;
            MinY = m_MinY; MaxY = m_MaxY;

            float AggX = 0, AggY = 0, LengthX = 0, LengthY = 0;
            float AggFontX = 0, AggFontY = 0;
            string StrY = "";

            SetBorders(gPB, MaxX, MinX, MaxY, MinY, Height, Width,
                ref AggX, ref AggY, ref LengthX, ref LengthY,
                ref AggFontX, ref AggFontY, ref StrY);

            System.Drawing.PointF myPt = new System.Drawing.PointF();
            myPt.X = 0;
            myPt.Y = 0;
            gPB.DrawString(m_YTitle, m_LabelFont, myBrush, myPt);

            myPt.X = Width - gPB.MeasureString(m_XTitle, m_LabelFont).Width;
            myPt.Y = Height - gPB.MeasureString(m_XTitle, m_LabelFont).Height;
            gPB.DrawString(m_XTitle, m_LabelFont, myBrush, myPt);

            DrawXAxis(100, MinY, MaxX, MinX, MaxY, MinY, AggX, AggY, AggFontX, AggFontY, gPB);
            DrawYAxis(MinX, /*MaxX, MinX,*/ MaxY, MinY, AggX, AggY, AggFontX, AggFontY, StrY, gPB);

            int n = m_VecX.GetLength(0);
            double[] SY_Vec = new double[n];
            int[] Ent = new int[n];
            for (i = 0; i < n; i++)
            {
                Ent[i] = 1;
                SY_Vec[i] = 1;
                if ((m_VecX[i] >= m_MinX) && (m_VecX[i] <= m_MaxX) &&
                    (m_VecY[i] >= m_MinY) && (m_VecY[i] <= m_MaxY))
                {
                    var thepen = new Pen(Hue((m_VecZ[i] - MinZ) / (MaxZ - MinZ)));
                    m_Marker.Draw(gPB, thepen, new SolidBrush(Hue((m_VecZ[i] - MinZ) / (MaxZ - MinZ))), m_MarkerSize, (float)AffineX(m_VecX[i]), (float)AffineY(m_VecY[i]));
                    if (m_ShowErrorBars)
                    {
                        gPB.DrawLine(thepen, AffineX(m_VecX[i]), AffineY(m_VecY[i] + m_VecDX[i]), AffineX(m_VecX[i]), AffineY(m_VecY[i] - m_VecDY[i]));
                        gPB.DrawLine(thepen, AffineX(m_VecX[i] - m_DX / 4), AffineY(m_VecY[i] + m_VecDX[i]), AffineX(m_VecX[i] + m_DX / 4), AffineY(m_VecY[i] + m_VecDX[i]));
                        gPB.DrawLine(thepen, AffineX(m_VecX[i] - m_DX / 4), AffineY(m_VecY[i] - m_VecDY[i]), AffineX(m_VecX[i] + m_DX / 4), AffineY(m_VecY[i] - m_VecDY[i]));
                    }
                }
            };

            BilinearHueGradient(gPB, Width - 32, (int)AffineY(MaxY), 16, (int)(AffineY(MinY) - AffineY(MaxY)), 1.0, 1.0, 0.0, 0.0);
            int[] zticks;
            DrawLEGOTicks(gPB, 2, MinZ, MaxZ, Width - 17, (int)AffineY(MinY), (AffineY(MinY) - AffineY(MaxY)) / (MaxZ - MinZ), 1.0, -20, out zticks);

            if (m_ScatterFit > 0) Fit_Plot_Scatter(m_VecX, m_VecY, Ent, SY_Vec,
                                      MaxX, MinX, MaxY, MinY, m_ScatterFit, gPB);

            if (m_FunctionOverlayed) OverlayFunction(MaxX, MinX, MaxY, MinY, gPB);
            //All'esterno
            m_PlottedX = (double[])m_VecX.Clone();
            m_PlottedY = (double[])m_VecY.Clone();
            m_PlottedSY = (double[])SY_Vec.Clone();

            DrawPanel(gPB, 2);

            return null;
        }

        public double [][] ArrowPlot(System.Drawing.Graphics gPB, int Width, int Height)
        {
            SetQuality(gPB);
            double MinX = 0, MaxX = 0, MedX = 0, RMSx = 0;
            double MinY = 0, MaxY = 0, MedY = 0, RMSy = 0;            
            int i = 0;

            //gPB.Clear(Color.White);

            if (m_SetXDefaultLimits || (!m_SetXDefaultLimits && m_MaxX <= m_MinX))
                Fitting.FindStatistics(m_VecX, ref m_MaxX, ref m_MinX, ref MedX, ref RMSx);
            if (m_SetYDefaultLimits || (!m_SetYDefaultLimits && m_MaxY <= m_MinY))
                Fitting.FindStatistics(m_VecY, ref m_MaxY, ref m_MinY, ref MedY, ref RMSy);

            m_FitPar = new double[1];
            m_ParDescr = new string[1];

            MinX = m_MinX; MaxX = m_MaxX;
            MinY = m_MinY; MaxY = m_MaxY;

            float AggX = 0, AggY = 0, LengthX = 0, LengthY = 0;
            float AggFontX = 0, AggFontY = 0;
            string StrY = "";

            SetBorders(gPB, MaxX, MinX, MaxY, MinY, Height, Width,
                ref AggX, ref AggY, ref LengthX, ref LengthY,
                ref AggFontX, ref AggFontY, ref StrY);

            System.Drawing.PointF myPt = new System.Drawing.PointF();
            myPt.X = 0;
            myPt.Y = 0;
            gPB.DrawString(m_YTitle, m_LabelFont, myBrush, myPt);

            myPt.X = Width - gPB.MeasureString(m_XTitle, m_LabelFont).Width;
            myPt.Y = Height - gPB.MeasureString(m_XTitle, m_LabelFont).Height;
            gPB.DrawString(m_XTitle, m_LabelFont, myBrush, myPt);

            Brush br = new SolidBrush(m_HistoColor);
            Pen pn = new Pen(m_HistoColor);
            float th;

            string sample = m_ArrowSample.ToString(m_PanelFormat, System.Globalization.CultureInfo.InvariantCulture);
            myPt.X = myPt.X - 50 - gPB.MeasureString(sample, m_LabelFont).Width;
            myPt.Y = Height - (th = gPB.MeasureString(sample, m_LabelFont).Height);
            gPB.DrawString(sample, m_LabelFont, myBrush, myPt);

            gPB.DrawLine(pn, myPt.X, myPt.Y + th / 2, myPt.X - (float)(ShrinkX(m_ArrowSample) * m_ArrowScale), myPt.Y + th / 2);

            DrawXAxis(100, MinY, MaxX, MinX, MaxY, MinY, AggX, AggY, AggFontX, AggFontY, gPB);
            DrawYAxis(MinX, MaxY, MinY, AggX, AggY, AggFontX, AggFontY, StrY, gPB);

            int n = m_VecX.GetLength(0);
            double[] SY_Vec = new double[n];
            int[] Ent = new int[n];
            for (i = 0; i < n; i++)
            {
                Ent[i] = 1;
                SY_Vec[i] = 1;
                if ((m_VecX[i] >= m_MinX) && (m_VecX[i] <= m_MaxX) &&
                    (m_VecY[i] >= m_MinY) && (m_VecY[i] <= m_MaxY))
                {
                    float sx = (float)AffineX(m_VecX[i]);
                    float sy = (float)AffineY(m_VecY[i]);
                    float fx = (float)(AffineX(m_VecX[i] + m_VecDX[i] * m_ArrowScale));
                    float fy = (float)(AffineY(m_VecY[i] + m_VecDY[i] * m_ArrowScale));
                    float dx = (float)(fx - sx);
                    float dy = (float)(fy - sy);
                    float lx = (float)(dx / Math.Sqrt(dx * dx + dy * dy));
                    float ly = (float)(dy / Math.Sqrt(dx * dx + dy * dy));
                    float ax = (float)(fx - m_ArrowSize * lx + 0.5 * m_ArrowSize * ly);
                    float ay = (float)(fy - m_ArrowSize * ly - 0.5 * m_ArrowSize * lx);
                    float bx = (float)(fx - m_ArrowSize * lx - 0.5 * m_ArrowSize * ly);
                    float by = (float)(fy - m_ArrowSize * ly + 0.5 * m_ArrowSize * lx);
                    gPB.FillEllipse(br, sx - ShrinkX((MaxX - MinX) / 200), sy - ShrinkY((MaxY - MinY) / 200), ShrinkX((MaxX - MinX) / 100), ShrinkY((MaxY - MinY) / 100));
                    gPB.DrawLine(pn, sx, sy, fx, fy);
                    gPB.DrawLine(pn, fx, fy, ax, ay);
                    gPB.DrawLine(pn, fx, fy, bx, by);
                }
            };

/*            BilinearHueGradient(gPB, Width - 32, (int)AffineY(MaxY), 16, (int)(AffineY(MinY) - AffineY(MaxY)), 1.0, 1.0, 0.0, 0.0);
            int[] zticks;
            DrawLEGOTicks(gPB, 2, MinZ, MaxZ, Width - 17, (int)AffineY(MinY), (AffineY(MinY) - AffineY(MaxY)) / (MaxZ - MinZ), 1.0, -20, out zticks);

            if (m_ScatterFit > 0) Fit_Plot_Scatter(m_VecX, m_VecY, Ent, SY_Vec,
                                      MaxX, MinX, MaxY, MinY, m_ScatterFit, gPB);
*/
            if (m_FunctionOverlayed) OverlayFunction(MaxX, MinX, MaxY, MinY, gPB);
/*  
            m_PlottedX = (double[])m_VecX.Clone();
            m_PlottedY = (double[])m_VecY.Clone();
            m_PlottedSY = (double[])SY_Vec.Clone();
*/
            DrawPanel(gPB, 2);

            return null;
        }

        public double[][] GreyLevelArea(System.Drawing.Graphics gPB, int Width, int Height)
		{
			double [] X_Mean, Y_Mean;
			double [,] Z_Mean, nZ_Mean;
			//System.Drawing.Color [,] zcol;
			double MinX =0, MaxY =0, MedX =0, RMSx =0, k=1;
			double MaxX =0, MinY =0, MedY =0, RMSy =0;
			double MaxZ =0, MinZ =0;
			int i=0, j;

			//m_lastplotpainted = LastPlot.GArea;

			//gPB.Clear(Color.White);

			Fitting.Prepare_2DCustom_Distribution(m_VecX, m_VecY, m_DX, m_DY, 
				out X_Mean,out Y_Mean, 
				out Z_Mean, out nZ_Mean);

			if (m_SetXDefaultLimits || (!m_SetXDefaultLimits && m_MaxX<=m_MinX) )
				Fitting.FindStatistics(X_Mean, ref m_MaxX, ref m_MinX, ref MedX, ref RMSx);
			if (m_SetYDefaultLimits || (!m_SetYDefaultLimits && m_MaxY<=m_MinY) )
				Fitting.FindStatistics(Y_Mean, ref m_MaxY, ref m_MinY, ref MedY, ref RMSy);

			//Pulisce i vecchi parametri
			m_FitPar = new double [1];
			m_ParDescr= new string [1];

			MinX=m_MinX; MaxX=m_MaxX; 
			MinY=m_MinY; MaxY=m_MaxY;

			MinX = MinX - m_DX/2;
			MaxX = MaxX + m_DX/2;
			MinY = MinY - m_DY/2;
			MaxY = MaxY + m_DY/2;

			float AggX=0, AggY=0, LengthX=0, LengthY=0;
			float AggFontX=0, AggFontY=0;
			string StrY="";

			SetBorders(gPB, MaxX, MinX, MaxY, MinY, Height, Width, 
				ref AggX, ref AggY, ref LengthX, ref LengthY,
				ref AggFontX, ref AggFontY, ref StrY);

			int n = Z_Mean.GetLength(0);
			int m = Z_Mean.GetLength(1);
			MaxZ = Z_Mean[0,0];
			MinZ = MaxZ;

            double[][] o_vals = new double[3][] { new double[n * m], new double[n * m], new double[n * m] };

			for(i=0;i<n;i++)
				for(j=0;j<m;j++)
				{
                    o_vals[0][i * m + j] = X_Mean[i];
                    o_vals[1][i * m + j] = Y_Mean[j];
                    o_vals[2][i * m + j] = Z_Mean[i, j];
					if (Z_Mean[i,j]>MaxZ) MaxZ=Z_Mean[i,j];
					if (Z_Mean[i,j]<MinZ) MinZ=Z_Mean[i,j];
				};

			byte c;
 			for(i=0;i<n;i++)
				for(j=0;j<m;j++)
				{	
					if (((X_Mean[i]-m_DX/2)>= MinX) && ((X_Mean[i]+m_DX/2)<= MaxX))
					{
						c = (byte)(255-(255*(Z_Mean[i,j]-MinZ)/(MaxZ-MinZ)));
						Brush tmpBrush = new SolidBrush(Color.FromArgb(c,c,c));
						gPB.FillRectangle(tmpBrush, AffineX(X_Mean[i] - m_DX/2), AffineY(Y_Mean[j] + m_DY/2), ShrinkX(m_DX), ShrinkY(m_DY)); 
					};
				};

			//Passaggio all'esterno
			m_MaxZ = MaxZ;
			m_MinZ = MinZ;

			System.Drawing.PointF myPt = new System.Drawing.PointF();
			myPt.X = 0;
			myPt.Y = 0;
			gPB.DrawString(m_YTitle, m_LabelFont, myBrush, myPt);

			myPt.X = Width - gPB.MeasureString(m_XTitle,m_LabelFont).Width;
			myPt.Y = Height -gPB.MeasureString(m_XTitle,m_LabelFont).Height;
			gPB.DrawString(m_XTitle, m_LabelFont, myBrush, myPt);

			DrawXAxis(X_Mean.GetLength(0), MinY, MaxX, MinX, MaxY, MinY, AggX, AggY, AggFontX, AggFontY, gPB);
			DrawYAxis(MinX, /*MaxX, MinX, */MaxY, MinY, AggX, AggY, AggFontX, AggFontY, StrY, gPB);

			if (m_FunctionOverlayed) OverlayFunction(MaxX, MinX, MaxY, MinY, gPB);

			//All'esterno
			m_PlottedX = (double[])X_Mean.Clone();
			m_PlottedY = (double[])Y_Mean.Clone();
			m_PlottedMatZ = (double[,])Z_Mean.Clone();

			DrawPanel(gPB, 2);

            return o_vals;
		}

		public double [][] GAreaValues(System.Drawing.Graphics gPB, int Width, int Height)
		{
            SetQuality(gPB);
			double [] X_Mean, Y_Mean;
			double [,] Z_Mean, nZ_Mean;
			double MinX =0, MaxY =0, MedX =0, RMSx =0;
			double MaxX =0, MinY =0, MedY =0, RMSy =0;
			double MaxZ =0, MinZ =0;
			int i=0, j;


			//gPB.Clear(Color.White);
/*
			Fitting.Prepare_2DCustom_Distribution(m_VecX, m_VecY, m_DX, m_DY, 
				out X_Mean,out Y_Mean, 
				out Z_Mean, out nZ_Mean);
*/
			if (m_SetXDefaultLimits || (!m_SetXDefaultLimits && m_MaxX<=m_MinX) )
				Fitting.FindStatistics(m_VecX, ref m_MaxX, ref m_MinX, ref MedX, ref RMSx);
			if (m_SetYDefaultLimits || (!m_SetYDefaultLimits && m_MaxY<=m_MinY) )
				Fitting.FindStatistics(m_VecY, ref m_MaxY, ref m_MinY, ref MedY, ref RMSy);

			//Pulisce i vecchi parametri
			m_FitPar = new double [1];
			m_ParDescr= new string [1];

			MinX=m_MinX; MaxX=m_MaxX; 
			MinY=m_MinY; MaxY=m_MaxY;

			MinX = MinX - m_DX/2;
			MaxX = MaxX + m_DX/2;
			MinY = MinY - m_DY/2;
			MaxY = MaxY + m_DY/2;

			float AggX=0, AggY=0, LengthX=0, LengthY=0;
			float AggFontX=0, AggFontY=0;
			string StrY="";

			SetBorders(gPB, MaxX, MinX, MaxY, MinY, Height, Width, 
				ref AggX, ref AggY, ref LengthX, ref LengthY,
				ref AggFontX, ref AggFontY, ref StrY);

			int n = m_MatZ.GetLength(0);
			int m = m_MatZ.GetLength(1);
			MaxZ = m_MatZ[0,0];
			MinZ = MaxZ;

            double[][] o_vals = new double[3][] { new double[n * m], new double[n * m], new double[n * m] };

			for(i=0;i<n;i++)
				for(j=0;j<m;j++)
				{
                    o_vals[0][i * m + j] = m_VecX[i];
                    o_vals[1][i * m + j] = m_VecY[j];
                    o_vals[2][i * m + j] = m_MatZ[i, j];
					if (m_MatZ[i,j]>MaxZ) MaxZ=m_MatZ[i,j];
					if (m_MatZ[i,j]<MinZ) MinZ=m_MatZ[i,j];
				};

			byte c;
			for(i=0;i<n;i++)
				for(j=0;j<m;j++)
				{	
					if (((m_VecX[i]-m_DX/2)>= MinX) && ((m_VecX[i]+m_DX/2)<= MaxX))
					{
						c = (byte)(255-(255*Math.Sqrt((m_MatZ[i,j]-MinZ)/(MaxZ-MinZ))));
						Brush tmpBrush = new SolidBrush(Color.FromArgb(c,c,c));
						gPB.FillRectangle(tmpBrush, AffineX(m_VecX[i] - m_DX/2), AffineY(m_VecY[j] + m_DY/2), ShrinkX(m_DX), ShrinkY(m_DY)); 
					};
				};

			System.Drawing.PointF myPt = new System.Drawing.PointF();
			myPt.X = 0;
			myPt.Y = 0;
			gPB.DrawString(m_YTitle, m_LabelFont, myBrush, myPt);

			myPt.X = Width - gPB.MeasureString(m_XTitle,m_LabelFont).Width;
			myPt.Y = Height -gPB.MeasureString(m_XTitle,m_LabelFont).Height;
			gPB.DrawString(m_XTitle, m_LabelFont, myBrush, myPt);

			DrawXAxis(m_VecX.GetLength(0), MinY, MaxX, MinX, MaxY, MinY, AggX, AggY, AggFontX, AggFontY, gPB);
			DrawYAxis(MinX, /*MaxX, MinX, */ MaxY, MinY, AggX, AggY, AggFontX, AggFontY, StrY, gPB);

			if (m_FunctionOverlayed) OverlayFunction(MaxX, MinX, MaxY, MinY, gPB);

			m_PlottedX = (double[])m_VecX.Clone();
			m_PlottedY = (double[])m_VecY.Clone();
			m_PlottedMatZ = (double[,])m_MatZ.Clone();

			DrawPanel(gPB, 2);

            return o_vals;
		}

        public double[][] SymbolArea(System.Drawing.Graphics gPB, int Width, int Height)
        {
            SetQuality(gPB);
            double[] X_Mean, Y_Mean;
            double[,] Z_Mean, nZ_Mean;
            //System.Drawing.Color [,] zcol;
            double MinX = 0, MaxY = 0, MedX = 0, RMSx = 0, k = 1;
            double MaxX = 0, MinY = 0, MedY = 0, RMSy = 0;
            double MaxZ = 0, MinZ = 0;
            int i = 0, j;

            //m_lastplotpainted = LastPlot.GArea;

            //gPB.Clear(Color.White);

            Fitting.Prepare_2DCustom_Distribution(m_VecX, m_VecY, m_DX, m_DY,
                out X_Mean, out Y_Mean,
                out Z_Mean, out nZ_Mean);

            if (m_SetXDefaultLimits || (!m_SetXDefaultLimits && m_MaxX <= m_MinX))
                Fitting.FindStatistics(X_Mean, ref m_MaxX, ref m_MinX, ref MedX, ref RMSx);
            if (m_SetYDefaultLimits || (!m_SetYDefaultLimits && m_MaxY <= m_MinY))
                Fitting.FindStatistics(Y_Mean, ref m_MaxY, ref m_MinY, ref MedY, ref RMSy);

            //Pulisce i vecchi parametri
            m_FitPar = new double[1];
            m_ParDescr = new string[1];

            MinX = m_MinX; MaxX = m_MaxX;
            MinY = m_MinY; MaxY = m_MaxY;
            /*
                        MinX = MinX - m_DX/2;
                        MaxX = MaxX + m_DX/2;
                        MinY = MinY - m_DY/2;
                        MaxY = MaxY + m_DY/2;
            */
            float AggX = 0, AggY = 0, LengthX = 0, LengthY = 0;
            float AggFontX = 0, AggFontY = 0;
            string StrY = "";

            SetBorders(gPB, MaxX, MinX, MaxY, MinY, Height, Width,
                ref AggX, ref AggY, ref LengthX, ref LengthY,
                ref AggFontX, ref AggFontY, ref StrY);

            int n = Z_Mean.GetLength(0);
            int m = Z_Mean.GetLength(1);
            MaxZ = Z_Mean[0, 0];
            MinZ = MaxZ;

            double[][] o_vals = new double[3][] { new double[n * m], new double[n * m], new double[n * m] };

            for (i = 0; i < n; i++)
                for (j = 0; j < m; j++)
                {
                    o_vals[0][i * m + j] = X_Mean[i];
                    o_vals[1][i * m + j] = Y_Mean[j];
                    o_vals[2][i * m + j] = Z_Mean[i, j];
                    if (Z_Mean[i, j] > MaxZ) MaxZ = Z_Mean[i, j];
                    if (Z_Mean[i, j] < MinZ) MinZ = Z_Mean[i, j];
                };

            if (MaxZ == MinZ)
            {
                MaxZ += Math.Abs(MinZ) * 0.1;
                MinZ -= Math.Abs(MinZ) * 0.1;
                if (MaxZ == MinZ)
                {
                    MaxZ += 1.0;
                    MinZ -= 1.0;
                }
            }

            for (i = 0; i < n; i++)
                if (X_Mean[i] >= m_MinX && X_Mean[i] <= m_MaxX)
                    for (j = 0; j < m; j++)
                        if (Y_Mean[j] >= m_MinY && Y_Mean[j] <= m_MaxY)
                            if (Z_Mean[i, j] > 0)
                                m_Marker.Draw(gPB, myPen, myBrush, (uint)Math.Ceiling(Math.Sqrt(Z_Mean[i, j] / MaxZ) * m_MarkerSize), (float)AffineX(X_Mean[i]), (float)AffineY(Y_Mean[j]));

            for (i = 0; i <= m_MarkerSize; i++)
            {
                double zs = ((float)i / (float)m_MarkerSize);
                m_Marker.Draw(gPB, myPen, myBrush, (uint)i, (float)(Width - 16), (float)(AffineY(MinY) + zs * zs * (AffineY(MaxY) - AffineY(MinY))));
            }

            //DrawYAxis(Width - 128,MaxZ, MinZ, AggX, AggY, AggFontX, AggFontY, "", gPB); 
            int[] zticks;
            DrawLEGOTicks(gPB, 2, MinZ, MaxZ, Width - 17, (int)AffineY(MinY), (AffineY(MinY) - AffineY(MaxY)) / (MaxZ - MinZ), 1.0, -20, out zticks);

            //Passaggio all'esterno
            m_MaxZ = MaxZ;
            m_MinZ = MinZ;

            System.Drawing.PointF myPt = new System.Drawing.PointF();
            myPt.X = 0;
            myPt.Y = 0;
            gPB.DrawString(m_YTitle, m_LabelFont, myBrush, myPt);

            myPt.X = Width - gPB.MeasureString(m_XTitle, m_LabelFont).Width;
            myPt.Y = Height - gPB.MeasureString(m_XTitle, m_LabelFont).Height;
            gPB.DrawString(m_XTitle, m_LabelFont, myBrush, myPt);

            DrawXAxis(X_Mean.GetLength(0), MinY, MaxX, MinX, MaxY, MinY, AggX, AggY, AggFontX, AggFontY, gPB);
            DrawYAxis(MinX, /*MaxX, MinX, */MaxY, MinY, AggX, AggY, AggFontX, AggFontY, StrY, gPB);

            if (m_FunctionOverlayed) OverlayFunction(MaxX, MinX, MaxY, MinY, gPB);

            //All'esterno
            m_PlottedX = (double[])X_Mean.Clone();
            m_PlottedY = (double[])Y_Mean.Clone();
            m_PlottedMatZ = (double[,])Z_Mean.Clone();

            DrawPanel(gPB, 2);

            return o_vals;
        }

		public double [][] HueArea(System.Drawing.Graphics gPB, int Width, int Height)
		{
            SetQuality(gPB);
			double [] X_Mean, Y_Mean;
			double [,] Z_Mean, nZ_Mean;
			//System.Drawing.Color [,] zcol;
			double MinX =0, MaxY =0, MedX =0, RMSx =0, k=1;
			double MaxX =0, MinY =0, MedY =0, RMSy =0;
			double MaxZ =0, MinZ =0;
			int i=0, j;

			//m_lastplotpainted = LastPlot.GArea;

			//gPB.Clear(Color.White);

			Fitting.Prepare_2DCustom_Distribution(m_VecX, m_VecY, m_DX, m_DY, 
				out X_Mean, out Y_Mean, 
				out Z_Mean, out nZ_Mean);

			if (m_SetXDefaultLimits || (!m_SetXDefaultLimits && m_MaxX<=m_MinX) )
				Fitting.FindStatistics(X_Mean, ref m_MaxX, ref m_MinX, ref MedX, ref RMSx);
			if (m_SetYDefaultLimits || (!m_SetYDefaultLimits && m_MaxY<=m_MinY) )
				Fitting.FindStatistics(Y_Mean, ref m_MaxY, ref m_MinY, ref MedY, ref RMSy);

			//Pulisce i vecchi parametri
			m_FitPar = new double [1];
			m_ParDescr= new string [1];

			MinX=m_MinX; MaxX=m_MaxX; 
			MinY=m_MinY; MaxY=m_MaxY;
/*
			MinX = MinX - m_DX/2;
			MaxX = MaxX + m_DX/2;
			MinY = MinY - m_DY/2;
			MaxY = MaxY + m_DY/2;
*/
			float AggX=0, AggY=0, LengthX=0, LengthY=0;
			float AggFontX=0, AggFontY=0;
			string StrY="";

			SetBorders(gPB, MaxX, MinX, MaxY, MinY, Height, Width, 
				ref AggX, ref AggY, ref LengthX, ref LengthY,
				ref AggFontX, ref AggFontY, ref StrY);

			int n = Z_Mean.GetLength(0);
			int m = Z_Mean.GetLength(1);
			MaxZ = Z_Mean[0,0];
			MinZ = MaxZ;

            double[][] o_vals = new double[3][] { new double[n * m], new double[n * m], new double[n * m] };

			for(i=0;i<n;i++)
				for(j=0;j<m;j++)
				{
                    o_vals[0][i * m + j] = X_Mean[i];
                    o_vals[1][i * m + j] = Y_Mean[j];
                    o_vals[2][i * m + j] = Z_Mean[i, j];
					if (Z_Mean[i,j]>MaxZ) MaxZ=Z_Mean[i,j];
					if (Z_Mean[i,j]<MinZ) MinZ=Z_Mean[i,j];
				};

            if (MaxZ == MinZ)
            {
                MaxZ += Math.Abs(MinZ) * 0.1;
                MinZ -= Math.Abs(MinZ) * 0.1;
                if (MaxZ == MinZ)
                {
                    MaxZ += 1.0;
                    MinZ -= 1.0;
                }
            }
			
			for(i = 0; i < n - 1; i++)
				if (X_Mean[i] >= m_MinX && X_Mean[i + 1] <= m_MaxX)
					for(j = 0; j < m - 1; j++)
						if (Y_Mean[j] >= m_MinY && Y_Mean[j + 1] <= m_MaxY)
						{	
/*
 							int x = (int)AffineX(X_Mean[i] - m_DX/2);
							int y = (int)AffineY(Y_Mean[j + 1] - m_DY/2);
							int w = (int)AffineX(X_Mean[i + 1] - m_DX/2) - x;
							int h = (int)AffineY(Y_Mean[j] - m_DY/2) - y;
*/							
							int x = (int)AffineX(X_Mean[i]);
							int y = (int)AffineY(Y_Mean[j + 1]);
							int w = (int)AffineX(X_Mean[i + 1]) - x;
							int h = (int)AffineY(Y_Mean[j]) - y;
							BilinearHueGradient(gPB, x, y, w, h, Math.Sqrt((Z_Mean[i,j + 1]-MinZ)/(MaxZ-MinZ)), Math.Sqrt((Z_Mean[i + 1,j + 1]-MinZ)/(MaxZ-MinZ)), Math.Sqrt((Z_Mean[i,j]-MinZ)/(MaxZ-MinZ)), Math.Sqrt((Z_Mean[i + 1,j]-MinZ)/(MaxZ-MinZ)));
						};

			//BilinearHueGradient(gPB, Width - 32, (int)AffineY(MaxY), 16, (int)AffineY(MinY) - (int)AffineY(MaxY), 1.0, 1.0, 0.0, 0.0); 
			for (i = (int)AffineY(MaxY); i < AffineY(MinY); i++)
			{
				double x = Math.Sqrt((AffineY(MinY) - i)/(AffineY(MinY) - AffineY(MaxY)));				
				BilinearHueGradient(gPB, Width - 32, i, 16, 1, x, x, x, x); 
			}

			//DrawYAxis(Width - 128,MaxZ, MinZ, AggX, AggY, AggFontX, AggFontY, "", gPB); 
			int [] zticks;
			DrawLEGOTicks(gPB, 2, MinZ, MaxZ, Width - 17, (int)AffineY(MinY), (AffineY(MinY)-AffineY(MaxY)) / (MaxZ - MinZ),  1.0, -20, out zticks);

			//Passaggio all'esterno
			m_MaxZ = MaxZ;
			m_MinZ = MinZ;

			System.Drawing.PointF myPt = new System.Drawing.PointF();
			myPt.X = 0;
			myPt.Y = 0;
			gPB.DrawString(m_YTitle, m_LabelFont, myBrush, myPt);

			myPt.X = Width - gPB.MeasureString(m_XTitle,m_LabelFont).Width;
			myPt.Y = Height -gPB.MeasureString(m_XTitle,m_LabelFont).Height;
			gPB.DrawString(m_XTitle, m_LabelFont, myBrush, myPt);

			DrawXAxis(X_Mean.GetLength(0), MinY, MaxX, MinX, MaxY, MinY, AggX, AggY, AggFontX, AggFontY, gPB);
			DrawYAxis(MinX, /*MaxX, MinX, */MaxY, MinY, AggX, AggY, AggFontX, AggFontY, StrY, gPB);

			if (m_FunctionOverlayed) OverlayFunction(MaxX, MinX, MaxY, MinY, gPB);

			//All'esterno
			m_PlottedX = (double[])X_Mean.Clone();
			m_PlottedY = (double[])Y_Mean.Clone();
			m_PlottedMatZ = (double[,])Z_Mean.Clone();

			DrawPanel(gPB, 2);

            return o_vals;
		}

		public double [][] HueAreaValues(System.Drawing.Graphics gPB, int Width, int Height)
		{
            SetQuality(gPB);
			double MinX =0, MaxY =0, MedX =0, RMSx =0;
			double MaxX =0, MinY =0, MedY =0, RMSy =0;
			double MaxZ =0, MinZ =0;
			int i=0, j;


			//gPB.Clear(Color.White);
			if (m_SetXDefaultLimits || (!m_SetXDefaultLimits && m_MaxX<=m_MinX) )
				Fitting.FindStatistics(m_VecX, ref m_MaxX, ref m_MinX, ref MedX, ref RMSx);
			if (m_SetYDefaultLimits || (!m_SetYDefaultLimits && m_MaxY<=m_MinY) )
				Fitting.FindStatistics(m_VecY, ref m_MaxY, ref m_MinY, ref MedY, ref RMSy);

			//Pulisce i vecchi parametri
			m_FitPar = new double [1];
			m_ParDescr= new string [1];

			MinX=m_MinX; MaxX=m_MaxX; 
			MinY=m_MinY; MaxY=m_MaxY;
/*
			MinX = MinX - m_DX/2;
			MaxX = MaxX + m_DX/2;
			MinY = MinY - m_DY/2;
			MaxY = MaxY + m_DY/2;
*/
			float AggX=0, AggY=0, LengthX=0, LengthY=0;
			float AggFontX=0, AggFontY=0;
			string StrY="";

			SetBorders(gPB, MaxX, MinX, MaxY, MinY, Height, Width, 
				ref AggX, ref AggY, ref LengthX, ref LengthY,
				ref AggFontX, ref AggFontY, ref StrY);

			int n = m_MatZ.GetLength(0);
			int m = m_MatZ.GetLength(1);
			MaxZ = m_MatZ[0,0];
			MinZ = MaxZ;

            double[][] o_vals = new double[3][] { new double[n * m], new double[n * m], new double[n * m] };

			for(i=0;i<n;i++)
				for(j=0;j<m;j++)
				{
                    o_vals[0][i * m + j] = m_VecX[i];
                    o_vals[1][i * m + j] = m_VecY[j];
                    o_vals[2][i * m + j] = m_MatZ[i, j];
					if (m_MatZ[i,j]>MaxZ) MaxZ=m_MatZ[i,j];
					if (m_MatZ[i,j]<MinZ) MinZ=m_MatZ[i,j];
				};

            if (MaxZ == MinZ)
            {
                MaxZ += Math.Abs(MinZ) * 0.1;
                MinZ -= Math.Abs(MinZ) * 0.1;
                if (MaxZ == MinZ)
                {
                    MaxZ += 1.0;
                    MinZ -= 1.0;
                }
            }

			for(i = 0; i < n - 1; i++)
				if (m_VecX[i] >= m_MinX && m_VecX[i + 1] <= m_MaxX)
					for(j = 0; j < m - 1; j++)
						if (m_VecY[j] >= m_MinY && m_VecY[j + 1] <= m_MaxY)
						{	
/*
							int x = (int)AffineX(m_VecX[i] - m_DX/2);
							int y = (int)AffineY(m_VecY[j] - m_DY/2);
							int w = (int)AffineX(m_VecX[i + 1] - m_DX/2) - x;
							int h = (int)AffineY(m_VecY[j + 1] - m_DY/2) - y;
*/							
							int x = (int)AffineX(m_VecX[i]);
							int y = (int)AffineY(m_VecY[j]);
							int w = (int)AffineX(m_VecX[i + 1]) - x;
							int h = (int)AffineY(m_VecY[j + 1]) - y;
							//BilinearHueGradient(gPB, x, y, w, h, (m_MatZ[i,j + 1]-MinZ)/(MaxZ-MinZ), (m_MatZ[i + 1,j + 1]-MinZ)/(MaxZ-MinZ), (m_MatZ[i,j]-MinZ)/(MaxZ-MinZ), (m_MatZ[i + 1,j]-MinZ)/(MaxZ-MinZ));
							BilinearHueGradient(gPB, x, y, w, h, Math.Sqrt((m_MatZ[i,j + 1]-MinZ)/(MaxZ-MinZ)), Math.Sqrt((m_MatZ[i + 1,j + 1]-MinZ)/(MaxZ-MinZ)), Math.Sqrt((m_MatZ[i,j]-MinZ)/(MaxZ-MinZ)), Math.Sqrt((m_MatZ[i + 1,j]-MinZ)/(MaxZ-MinZ)));
						};

            BilinearHueGradient(gPB, Width - 32, (int)AffineY(MaxY), 16, (int)(AffineY(MinY) - AffineY(MaxY)), 1.0, 1.0, 0.0, 0.0); 

			System.Drawing.PointF myPt = new System.Drawing.PointF();
			myPt.X = 0;
			myPt.Y = 0;
			gPB.DrawString(m_YTitle, m_LabelFont, myBrush, myPt);

			myPt.X = Width - gPB.MeasureString(m_XTitle,m_LabelFont).Width;
			myPt.Y = Height -gPB.MeasureString(m_XTitle,m_LabelFont).Height;
			gPB.DrawString(m_XTitle, m_LabelFont, myBrush, myPt);

			DrawXAxis(m_VecX.GetLength(0), MinY, MaxX, MinX, MaxY, MinY, AggX, AggY, AggFontX, AggFontY, gPB);
			DrawYAxis(MinX, /*MaxX, MinX, */ MaxY, MinY, AggX, AggY, AggFontX, AggFontY, StrY, gPB);

			if (m_FunctionOverlayed) OverlayFunction(MaxX, MinX, MaxY, MinY, gPB);

			m_PlottedX = (double[])m_VecX.Clone();
			m_PlottedY = (double[])m_VecY.Clone();
			m_PlottedMatZ = (double[,])m_MatZ.Clone();

			DrawPanel(gPB, 2);

            return o_vals;
		}

		public double [][] HueAreaComputedValues(System.Drawing.Graphics gPB, int Width, int Height)
		{
            SetQuality(gPB);
			double MinX =0, MaxY =0, MedX =0, RMSx =0;
			double MaxX =0, MinY =0, MedY =0, RMSy =0;
			double MaxZ =0, MinZ =0;
			double[] X_Mean, Y_Mean; 
			double [,] Z_Mean, rmsZ_Mean;
			int [,] nEnt;
			int i=0, j;


			//gPB.Clear(Color.White);

			Fitting.Prepare_2DCustom_Distribution_ZVal(m_VecX, m_VecY, m_VecZ, m_DX, m_DY, 
				out X_Mean,out Y_Mean, 
				out Z_Mean, out rmsZ_Mean, out nEnt);

			if (m_SetXDefaultLimits || (!m_SetXDefaultLimits && m_MaxX<=m_MinX) )
				Fitting.FindStatistics(X_Mean, ref m_MaxX, ref m_MinX, ref MedX, ref RMSx);
			if (m_SetYDefaultLimits || (!m_SetYDefaultLimits && m_MaxY<=m_MinY) )
				Fitting.FindStatistics(Y_Mean, ref m_MaxY, ref m_MinY, ref MedY, ref RMSy);

			//Pulisce i vecchi parametri
			m_FitPar = new double [1];
			m_ParDescr= new string [1];

			MinX=m_MinX; MaxX=m_MaxX; 
			MinY=m_MinY; MaxY=m_MaxY;
/*
			MinX = MinX - m_DX/2;
			MaxX = MaxX + m_DX/2;
			MinY = MinY - m_DY/2;
			MaxY = MaxY + m_DY/2;
*/
			float AggX=0, AggY=0, LengthX=0, LengthY=0;
			float AggFontX=0, AggFontY=0;
			string StrY="";

			SetBorders(gPB, MaxX, MinX, MaxY, MinY, Height, Width, 
				ref AggX, ref AggY, ref LengthX, ref LengthY,
				ref AggFontX, ref AggFontY, ref StrY);

			int n = Z_Mean.GetLength(0);
			int m = Z_Mean.GetLength(1);
			MaxZ = Z_Mean[0,0];
			MinZ = MaxZ;

            double[][] o_vals = new double[4][] { new double[n * m], new double[n * m], new double[n * m], new double[n * m] };

			for(i=0;i<n;i++)
				for(j=0;j<m;j++)
				{
                    o_vals[0][i * m + j] = X_Mean[i];
                    o_vals[1][i * m + j] = Y_Mean[j];
                    o_vals[2][i * m + j] = Z_Mean[i, j];
                    o_vals[3][i * m + j] = nEnt[i, j];
					if (Z_Mean[i,j]>MaxZ) MaxZ=Z_Mean[i,j];
					if (Z_Mean[i,j]<MinZ) MinZ=Z_Mean[i,j];
				};

            if (MaxZ == MinZ)
            {
                MaxZ += Math.Abs(MinZ) * 0.1;
                MinZ -= Math.Abs(MinZ) * 0.1;
                if (MaxZ == MinZ)
                {
                    MaxZ += 1.0;
                    MinZ -= 1.0;
                }
            }

			for(i = 0; i < n - 1; i++)
				for(j = 0; j < m - 1; j++)
				{	
/*
					int x = (int)AffineX(X_Mean[i] - m_DX/2);
					int y = (int)AffineY(Y_Mean[j + 1] - m_DY/2);
					int w = (int)AffineX(X_Mean[i + 1] - m_DX/2) - x;
					int h = (int)AffineY(Y_Mean[j] - m_DY/2) - y;
*/					
					int x = (int)AffineX(X_Mean[i]);
					int y = (int)AffineY(Y_Mean[j + 1]);
					int w = (int)AffineX(X_Mean[i + 1]) - x;
					int h = (int)AffineY(Y_Mean[j]) - y;
					BilinearHueGradient(gPB, x, y, w, h, (Z_Mean[i,j + 1]-MinZ)/(MaxZ-MinZ), (Z_Mean[i + 1,j + 1]-MinZ)/(MaxZ-MinZ), (Z_Mean[i,j]-MinZ)/(MaxZ-MinZ), (Z_Mean[i + 1,j]-MinZ)/(MaxZ-MinZ),
						nEnt[i, j + 1] > 0, nEnt[i + 1, j + 1] > 0, nEnt[i, j] > 0, nEnt[i + 1, j] > 0);
				};

			BilinearHueGradient(gPB, Width - 32, (int)AffineY(MaxY), 16, (int)AffineY(MinY) - (int)AffineY(MaxY), 1.0, 1.0, 0.0, 0.0); 
			int [] zticks;
			DrawLEGOTicks(gPB, 3, MinZ, MaxZ, Width - 17, (int)AffineY(MinY), (AffineY(MinY)-AffineY(MaxY)) / (MaxZ - MinZ),  1.0, -20, out zticks);

			//Passaggio all'esterno
			m_MaxZ = MaxZ;
			m_MinZ = MinZ;

			System.Drawing.PointF myPt = new System.Drawing.PointF();
			myPt.X = 0;
			myPt.Y = 0;
			gPB.DrawString(m_YTitle, m_LabelFont, myBrush, myPt);

			myPt.X = Width - gPB.MeasureString(m_XTitle,m_LabelFont).Width;
			myPt.Y = Height -gPB.MeasureString(m_XTitle,m_LabelFont).Height;
			gPB.DrawString(m_XTitle, m_LabelFont, myBrush, myPt);

			DrawXAxis(X_Mean.GetLength(0), MinY, MaxX, MinX, MaxY, MinY, AggX, AggY, AggFontX, AggFontY, gPB);
			DrawYAxis(MinX, /*MaxX, MinX,*/ MaxY, MinY, AggX, AggY, AggFontX, AggFontY, StrY, gPB);

			if (m_FunctionOverlayed) OverlayFunction(MaxX, MinX, MaxY, MinY, gPB);

			m_PlottedX = (double[])X_Mean.Clone();
			m_PlottedY = (double[])Y_Mean.Clone();
			m_PlottedMatZ = (double[,])Z_Mean.Clone();

			DrawPanel(gPB, 2);

            return o_vals;
		}

		private double m_ObservationAngle = 30.0;

		private double m_ThreeDPlotC = 1.0, m_ThreeDPlotS = 0.0;
		private double m_ThreeDPlotXRescale = 1.0, m_ThreeDPlotYRescale = 1.0;

		public double ObservationAngle
		{
			get { return m_ObservationAngle; }

			set 
			{ 
				m_ObservationAngle = value - Math.Floor(value / 360.0) * 360; 
			}
		}

		private double m_Skewedness = 0.3;

		public double Skewedness
		{
			get { return m_Skewedness; }

			set { m_Skewedness = value; }
		}

        private double m_ArrowScale = 1.0;

        public double ArrowScale
        {
            get { return m_ArrowScale; }

            set { m_ArrowScale = value; }
        }

        private double m_ArrowSample = 1.0;

        public double ArrowSample
        {
            get { return m_ArrowSample; }

            set { m_ArrowSample = value; }
        }

        private double m_ArrowSize = 5.0;

        public double ArrowSize
        {
            get { return m_ArrowSize; }

            set { m_ArrowSize = value; }
        }

        private void ThreeDTransform(double x, double y, out double tx, out double ty)
		{
			tx = x * m_ThreeDPlotXRescale * m_ThreeDPlotC - m_ThreeDPlotS * y * m_ThreeDPlotYRescale;
			ty = m_Skewedness * (x * m_ThreeDPlotXRescale * m_ThreeDPlotS + y * m_ThreeDPlotYRescale * m_ThreeDPlotC);
		}

		public double [][] LEGOPlot(System.Drawing.Graphics gPB, int Width, int Height)
		{
            SetQuality(gPB);
			const int XYTickLength = 8;
			const int ZTickLength = 4;
			int [] zticklevels = null;
			int zlevel;

			double [] X_Mean, Y_Mean;
			double [,] Z_Mean, nZ_Mean;
			double XRange, YRange;
		
			double X = 0, Y = 0;
			double MinX =0, MaxY =0;
			double MaxX =0, MinY =0;
			double MaxZ =0;
			int i, i0, i1;

			m_ThreeDPlotC = Math.Cos(m_ObservationAngle / 180.0 * Math.PI);
			m_ThreeDPlotS = Math.Sin(m_ObservationAngle / 180.0 * Math.PI);			

			//gPB.Clear(Color.White);

			if (m_SetXDefaultLimits || (!m_SetXDefaultLimits && m_MaxX<=m_MinX) )
				Fitting.FindStatistics(m_VecX, ref m_MaxX, ref m_MinX, ref X, ref Y);
			if (m_SetYDefaultLimits || (!m_SetYDefaultLimits && m_MaxY<=m_MinY) )
				Fitting.FindStatistics(m_VecY, ref m_MaxY, ref m_MinY, ref X, ref Y);
			Fitting.Prepare_2DCustom_Distribution(m_VecX, m_VecY, m_DX, m_DY, 
				m_MinX, m_MaxX, m_MinY, m_MaxY,
				out X_Mean,out Y_Mean, 
				out Z_Mean, out nZ_Mean);

			MinX = m_MinX; MaxX = m_MaxX; 
			MinY = m_MinY; MaxY = m_MaxY;

			MinX = Math.Floor(MinX / m_DX) * m_DX - m_DX/2;
			MaxX = Math.Ceiling(MaxX / m_DX) * m_DX + m_DX/2;
			XRange = MaxX - MinX; m_ThreeDPlotXRescale = 1.0 / XRange;
			MinY = Math.Floor(MinY / m_DY) * m_DY - m_DY/2;
			MaxY = Math.Ceiling(MaxY / m_DY) * m_DY + m_DY/2;
			YRange = MaxY - MinY; m_ThreeDPlotYRescale = 1.0 / YRange;

			if (MaxX <= MinX) MaxX = 1.0 + MinX;
			if (MaxY <= MinY) MaxY = 1.0 + MinY;

			double XScale, YScale, ZScale;
			double MinXT, MaxXT, MinYT, MaxYT;

			ThreeDTransform(MinX, MinY, out MinXT, out MinYT);
			MaxXT = MinXT;
			MaxYT = MinYT;
			ThreeDTransform(MaxX, MinY, out X, out Y);
			if (X < MinXT) MinXT = X;
			if (X > MaxXT) MaxXT = X;
			if (Y < MinYT) MinYT = Y;
			if (Y > MaxYT) MaxYT = Y;
			ThreeDTransform(MinX, MaxY, out X, out Y);
			if (X < MinXT) MinXT = X;
			if (X > MaxXT) MaxXT = X;
			if (Y < MinYT) MinYT = Y;
			if (Y > MaxYT) MaxYT = Y;
			ThreeDTransform(MaxX, MaxY, out X, out Y);
			if (X < MinXT) MinXT = X;
			if (X > MaxXT) MaxXT = X;
			if (Y < MinYT) MinYT = Y;
			if (Y > MaxYT) MaxYT = Y;

			int n0 = Z_Mean.GetLength(0);
			int n1 = Z_Mean.GetLength(1);
			MaxZ = 0;

            double[][] o_vals = new double[3][] { new double[n0 * n1], new double[n0 * n1], new double[n0 * n1] };

			for(i0 = 0; i0 < n0; i0++)
                for (i1 = 0; i1 < n1; i1++)
                {
                    o_vals[0][i0 * n1 + i1] = MinX + i0 * m_DX;
                    o_vals[1][i0 * n1 + i1] = MinY + i1 * m_DY;
                    o_vals[2][i0 * n1 + i1] = Z_Mean[i0, i1];
                    if (Z_Mean[i0, i1] > MaxZ) MaxZ = Z_Mean[i0, i1];
                }

			int xprec = (int)Math.Ceiling(-Math.Log10(m_DX)); if (xprec < 0) xprec = 0;
			int yprec = (int)Math.Ceiling(-Math.Log10(m_DY)); if (yprec < 0) yprec = 0;
			
			int XSizeOfTitles = 2 * (int)Math.Max(gPB.MeasureString(MaxZ.ToString(), m_LabelFont).Width + Math.Abs(ZTickLength), Math.Max(
				Math.Max(gPB.MeasureString(MinX.ToString("F" + xprec), m_LabelFont).Width + Math.Abs(XYTickLength), gPB.MeasureString(MaxX.ToString("F" + xprec), m_LabelFont).Width + Math.Abs(XYTickLength)), 
				Math.Max(gPB.MeasureString(MinY.ToString("F" + xprec), m_LabelFont).Width + Math.Abs(XYTickLength), gPB.MeasureString(MaxY.ToString("F" + yprec), m_LabelFont).Width + Math.Abs(XYTickLength))
				));			

			if (MaxXT > MinXT) XScale = (Width - XSizeOfTitles) / (MaxXT - MinXT);
			else XScale = 1.0;

			YScale = (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) * m_Skewedness / (MaxYT - MinYT);
			ZScale = (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) * (1.0 - m_Skewedness) / MaxZ;

			Brush XSideBrush = new System.Drawing.SolidBrush(Color.FromArgb(224, 224, 224));
			Brush YSideBrush = new System.Drawing.SolidBrush(Color.FromArgb(192, 192, 192));
			Brush TopBrush = new System.Drawing.SolidBrush(Color.FromArgb(224, 0, 0));
			Pen LevelPen = new Pen(Color.FromArgb(128, 128, 192), 1);

			int [,] xb = new int[n0 + 1, n1 + 1];
			int [,] yb = new int[n0 + 1, n1 + 1];

			for(i0 = 0; i0 <= n0; i0++)
				for(i1 = 0; i1 <= n1; i1++)
				{
					ThreeDTransform(MinX + i0 * m_DX, MinY + i1 * m_DY, out X, out Y);
					xb[i0, i1] = (int)((X - MinXT) * XScale) + XSizeOfTitles / 2;
					yb[i0, i1] = m_LabelFont.Height + (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) - (int)((Y - MinYT) * YScale);
				}

			Point [] xsidepts = new Point[4];
			Point [] ysidepts = new Point[4];
			Point [] toppts = new Point[4];
			Point [] topboxpts = new Point[4];
			Point [] bottomboxpts = new Point[4];
			bottomboxpts[0].X = topboxpts[0].X = xb[0, 0];
			bottomboxpts[0].Y = yb[0, 0] + 1;
			bottomboxpts[1].X = topboxpts[1].X = xb[n0, 0];
			bottomboxpts[1].Y = yb[n0, 0] + 1;
			bottomboxpts[2].X = topboxpts[2].X = xb[n0, n1];
			bottomboxpts[2].Y = yb[n0, n1] + 1;
			bottomboxpts[3].X = topboxpts[3].X = xb[0, n1];
			bottomboxpts[3].Y = yb[0, n1] + 1;
			gPB.DrawPolygon(myPen, bottomboxpts);
			topboxpts[0].Y = bottomboxpts[0].Y - (int)(MaxZ * ZScale) - 1;
			topboxpts[1].Y = bottomboxpts[1].Y - (int)(MaxZ * ZScale) - 1;
			topboxpts[2].Y = bottomboxpts[2].Y - (int)(MaxZ * ZScale) - 1;
			topboxpts[3].Y = bottomboxpts[3].Y - (int)(MaxZ * ZScale) - 1;
			switch ((int)(m_ObservationAngle / 45.0))
			{
				case 0:
					DrawLEGOTicks(gPB, 0, MinX, MaxX, xb[0, 0], yb[0, 0], XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 1, MinY, MaxY, xb[0, 0], yb[0, 0], XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 2, 0, MaxZ, xb[0, n1], yb[0, n1], ZScale, ZScale, -ZTickLength, out zticklevels);
					gPB.DrawLine(myPen, bottomboxpts[2], topboxpts[2]);
					gPB.DrawLine(myPen, bottomboxpts[3], topboxpts[3]);
					for (i1 = n1 - 1; i1 >= 0; i1--)
						for (i0 = n0 - 1; i0 >= 0; i0--)
						{
							toppts[0].X = ysidepts[0].X = ysidepts[1].X = xb[i0, i1 + 1];
							ysidepts[0].Y = yb[i0, i1 + 1];
							zlevel =  - (int)(Z_Mean[i0, i1] * ZScale);
							toppts[0].Y = ysidepts[1].Y = yb[i0, i1 + 1] + zlevel;
							toppts[1].X = ysidepts[2].X = ysidepts[3].X = xb[i0, i1];
							ysidepts[3].Y = yb[i0, i1];
							toppts[1].Y = ysidepts[2].Y = yb[i0, i1] + zlevel;
							gPB.FillPolygon(YSideBrush, ysidepts);
							xsidepts[0].X = xsidepts[1].X = xb[i0, i1];
							xsidepts[0].Y = yb[i0, i1];
							xsidepts[1].Y = yb[i0, i1] + zlevel;
							toppts[2].X = xsidepts[2].X = xsidepts[3].X = xb[i0 + 1, i1];
							xsidepts[3].Y = yb[i0 + 1, i1];
							toppts[2].Y = xsidepts[2].Y = yb[i0 + 1, i1] + zlevel;
							gPB.FillPolygon(XSideBrush, xsidepts);
							toppts[3].X = xb[i0 + 1, i1 + 1];
							toppts[3].Y = yb[i0 + 1, i1 + 1] + zlevel;
							gPB.FillPolygon(TopBrush, toppts);
							for (i = 0; i < zticklevels.Length && zticklevels[i] >= zlevel; i++)
							{
								gPB.DrawLine(LevelPen, ysidepts[0].X, ysidepts[0].Y + zticklevels[i], ysidepts[3].X, ysidepts[3].Y + zticklevels[i]);
								gPB.DrawLine(LevelPen, xsidepts[0].X, xsidepts[0].Y + zticklevels[i], xsidepts[3].X, xsidepts[3].Y + zticklevels[i]);
							}
						}							
					gPB.DrawLine(myPen, bottomboxpts[0], topboxpts[0]);
					gPB.DrawLine(myPen, bottomboxpts[1], topboxpts[1]);
					break;

				case 1:
					DrawLEGOTicks(gPB, 0, MinX, MaxX, xb[0, 0], yb[0, 0], XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 1, MinY, MaxY, xb[0, 0], yb[0, 0], XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 2, 0, MaxZ, xb[0, n1], yb[0, n1], ZScale, ZScale, -ZTickLength, out zticklevels);
					gPB.DrawLine(myPen, bottomboxpts[2], topboxpts[2]);
					gPB.DrawLine(myPen, bottomboxpts[3], topboxpts[3]);
					for (i0 = n0 - 1; i0 >= 0; i0--)
						for (i1 = n1 - 1; i1 >= 0; i1--)						
						{
							toppts[0].X = ysidepts[0].X = ysidepts[1].X = xb[i0, i1 + 1];
							ysidepts[0].Y = yb[i0, i1 + 1];
							zlevel =  - (int)(Z_Mean[i0, i1] * ZScale);
							toppts[0].Y = ysidepts[1].Y = yb[i0, i1 + 1] + zlevel;
							toppts[1].X = ysidepts[2].X = ysidepts[3].X = xb[i0, i1];
							ysidepts[3].Y = yb[i0, i1];
							toppts[1].Y = ysidepts[2].Y = yb[i0, i1] + zlevel;
							gPB.FillPolygon(YSideBrush, ysidepts);
							xsidepts[0].X = xsidepts[1].X = xb[i0, i1];
							xsidepts[0].Y = yb[i0, i1];
							xsidepts[1].Y = yb[i0, i1] + zlevel;
							toppts[2].X = xsidepts[2].X = xsidepts[3].X = xb[i0 + 1, i1];
							xsidepts[3].Y = yb[i0 + 1, i1];
							toppts[2].Y = xsidepts[2].Y = yb[i0 + 1, i1] + zlevel;
							gPB.FillPolygon(XSideBrush, xsidepts);
							toppts[3].X = xb[i0 + 1, i1 + 1];
							toppts[3].Y = yb[i0 + 1, i1 + 1] + zlevel;
							gPB.FillPolygon(TopBrush, toppts);
							for (i = 0; i < zticklevels.Length && zticklevels[i] >= zlevel; i++)
							{
								gPB.DrawLine(LevelPen, ysidepts[0].X, ysidepts[0].Y + zticklevels[i], ysidepts[3].X, ysidepts[3].Y + zticklevels[i]);
								gPB.DrawLine(LevelPen, xsidepts[0].X, xsidepts[0].Y + zticklevels[i], xsidepts[3].X, xsidepts[3].Y + zticklevels[i]);
							}
						}							
					gPB.DrawLine(myPen, bottomboxpts[0], topboxpts[0]);
					gPB.DrawLine(myPen, bottomboxpts[1], topboxpts[1]);
					break;

				case 2:
					DrawLEGOTicks(gPB, 0, MinX, MaxX, xb[0, n1], yb[0, n1], XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 1, MinY, MaxY, xb[0, 0], yb[0, 0], XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 2, 0, MaxZ, xb[n0, n1], yb[n0, n1], ZScale, ZScale, -ZTickLength, out zticklevels);
					gPB.DrawLine(myPen, bottomboxpts[0], topboxpts[0]);
					gPB.DrawLine(myPen, bottomboxpts[1], topboxpts[1]);
					for (i1 = 0; i1 < n1; i1++)
						for (i0 = n0 - 1; i0 >= 0; i0--)
						{
							toppts[0].X = ysidepts[0].X = ysidepts[1].X = xb[i0, i1];
							ysidepts[0].Y = yb[i0, i1];
							zlevel =  - (int)(Z_Mean[i0, i1] * ZScale);
							toppts[0].Y = ysidepts[1].Y = yb[i0, i1] + zlevel;
							toppts[1].X = ysidepts[2].X = ysidepts[3].X = xb[i0, i1 + 1];
							ysidepts[3].Y = yb[i0, i1 + 1];
							toppts[1].Y = ysidepts[2].Y = yb[i0, i1 + 1] + zlevel;
							gPB.FillPolygon(YSideBrush, ysidepts);
							xsidepts[0].X = xsidepts[1].X = xb[i0, i1 + 1];
							xsidepts[0].Y = yb[i0, i1 + 1];
							xsidepts[1].Y = yb[i0, i1 + 1] + zlevel;
							toppts[2].X = xsidepts[2].X = xsidepts[3].X = xb[i0 + 1, i1 + 1];
							xsidepts[3].Y = yb[i0 + 1, i1 + 1];
							toppts[2].Y = xsidepts[2].Y = yb[i0 + 1, i1 + 1] + zlevel;
							gPB.FillPolygon(XSideBrush, xsidepts);
							toppts[3].X = xb[i0 + 1, i1];
							toppts[3].Y = yb[i0 + 1, i1] + zlevel;
							gPB.FillPolygon(TopBrush, toppts);
							for (i = 0; i < zticklevels.Length && zticklevels[i] >= zlevel; i++)
							{
								gPB.DrawLine(LevelPen, ysidepts[0].X, ysidepts[0].Y + zticklevels[i], ysidepts[3].X, ysidepts[3].Y + zticklevels[i]);
								gPB.DrawLine(LevelPen, xsidepts[0].X, xsidepts[0].Y + zticklevels[i], xsidepts[3].X, xsidepts[3].Y + zticklevels[i]);
							}
						}							
					gPB.DrawLine(myPen, bottomboxpts[2], topboxpts[2]);
					gPB.DrawLine(myPen, bottomboxpts[3], topboxpts[3]);
					break;

				case 3:
					DrawLEGOTicks(gPB, 0, MinX, MaxX, xb[0, n1], yb[0, n1], XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 1, MinY, MaxY, xb[0, 0], yb[0, 0], XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 2, 0, MaxZ, xb[n0, n1], yb[n0, n1], ZScale, ZScale, -ZTickLength, out zticklevels);
					gPB.DrawLine(myPen, bottomboxpts[0], topboxpts[0]);
					gPB.DrawLine(myPen, bottomboxpts[1], topboxpts[1]);
					for (i0 = n0 - 1; i0 >= 0; i0--)
						for (i1 = 0; i1 < n1; i1++)
						{
							toppts[0].X = ysidepts[0].X = ysidepts[1].X = xb[i0, i1];
							ysidepts[0].Y = yb[i0, i1];
							zlevel =  - (int)(Z_Mean[i0, i1] * ZScale);
							toppts[0].Y = ysidepts[1].Y = yb[i0, i1] + zlevel;
							toppts[1].X = ysidepts[2].X = ysidepts[3].X = xb[i0, i1 + 1];
							ysidepts[3].Y = yb[i0, i1 + 1];
							toppts[1].Y = ysidepts[2].Y = yb[i0, i1 + 1] + zlevel;
							gPB.FillPolygon(YSideBrush, ysidepts);
							xsidepts[0].X = xsidepts[1].X = xb[i0, i1 + 1];
							xsidepts[0].Y = yb[i0, i1 + 1];
							xsidepts[1].Y = yb[i0, i1 + 1] + zlevel;
							toppts[2].X = xsidepts[2].X = xsidepts[3].X = xb[i0 + 1, i1 + 1];
							xsidepts[3].Y = yb[i0 + 1, i1 + 1];
							toppts[2].Y = xsidepts[2].Y = yb[i0 + 1, i1 + 1] + zlevel;
							gPB.FillPolygon(XSideBrush, xsidepts);
							toppts[3].X = xb[i0 + 1, i1];
							toppts[3].Y = yb[i0 + 1, i1] + zlevel;
							gPB.FillPolygon(TopBrush, toppts);
							for (i = 0; i < zticklevels.Length && zticklevels[i] >= zlevel; i++)
							{
								gPB.DrawLine(LevelPen, ysidepts[0].X, ysidepts[0].Y + zticklevels[i], ysidepts[3].X, ysidepts[3].Y + zticklevels[i]);
								gPB.DrawLine(LevelPen, xsidepts[0].X, xsidepts[0].Y + zticklevels[i], xsidepts[3].X, xsidepts[3].Y + zticklevels[i]);
							}
						}							
					gPB.DrawLine(myPen, bottomboxpts[2], topboxpts[2]);
					gPB.DrawLine(myPen, bottomboxpts[3], topboxpts[3]);
					break;

				case 4:
					DrawLEGOTicks(gPB, 0, MinX, MaxX, xb[0, n1], yb[0, n1], XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 1, MinY, MaxY, xb[n0, 0], yb[n0, 0], XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 2, 0, MaxZ, xb[n0, 0], yb[n0, 0], ZScale, ZScale, -ZTickLength, out zticklevels);
					gPB.DrawLine(myPen, bottomboxpts[0], topboxpts[0]);
					gPB.DrawLine(myPen, bottomboxpts[1], topboxpts[1]);
					for (i1 = 0; i1 < n1; i1++)
						for (i0 = 0; i0 < n0; i0++)
						{
							toppts[0].X = ysidepts[0].X = ysidepts[1].X = xb[i0 + 1, i1];
							ysidepts[0].Y = yb[i0 + 1, i1];
							zlevel =  - (int)(Z_Mean[i0, i1] * ZScale);
							toppts[0].Y = ysidepts[1].Y = yb[i0 + 1, i1] + zlevel;
							toppts[1].X = ysidepts[2].X = ysidepts[3].X = xb[i0 + 1, i1 + 1];
							ysidepts[3].Y = yb[i0 + 1, i1 + 1];
							toppts[1].Y = ysidepts[2].Y = yb[i0 + 1, i1 + 1] + zlevel;
							gPB.FillPolygon(YSideBrush, ysidepts);
							xsidepts[0].X = xsidepts[1].X = xb[i0 + 1, i1 + 1];
							xsidepts[0].Y = yb[i0 + 1, i1 + 1];
							xsidepts[1].Y = yb[i0 + 1, i1 + 1] + zlevel;
							toppts[2].X = xsidepts[2].X = xsidepts[3].X = xb[i0, i1 + 1];
							xsidepts[3].Y = yb[i0, i1 + 1];
							toppts[2].Y = xsidepts[2].Y = yb[i0, i1 + 1] + zlevel;
							gPB.FillPolygon(XSideBrush, xsidepts);
							toppts[3].X = xb[i0, i1];
							toppts[3].Y = yb[i0, i1] + zlevel;
							gPB.FillPolygon(TopBrush, toppts);
							for (i = 0; i < zticklevels.Length && zticklevels[i] >= zlevel; i++)
							{
								gPB.DrawLine(LevelPen, ysidepts[0].X, ysidepts[0].Y + zticklevels[i], ysidepts[3].X, ysidepts[3].Y + zticklevels[i]);
								gPB.DrawLine(LevelPen, xsidepts[0].X, xsidepts[0].Y + zticklevels[i], xsidepts[3].X, xsidepts[3].Y + zticklevels[i]);
							}
						}							
					gPB.DrawLine(myPen, bottomboxpts[2], topboxpts[2]);
					gPB.DrawLine(myPen, bottomboxpts[3], topboxpts[3]);
					break;

				case 5:
					DrawLEGOTicks(gPB, 0, MinX, MaxX, xb[0, n1], yb[0, n1], XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 1, MinY, MaxY, xb[n0, 0], yb[n0, 0], XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 2, 0, MaxZ, xb[n0, 0], yb[n0, 0], ZScale, ZScale, -ZTickLength, out zticklevels);
					gPB.DrawLine(myPen, bottomboxpts[0], topboxpts[0]);
					gPB.DrawLine(myPen, bottomboxpts[1], topboxpts[1]);
					for (i0 = 0; i0 < n0; i0++)
						for (i1 = 0; i1 < n1; i1++)						
						{
							toppts[0].X = ysidepts[0].X = ysidepts[1].X = xb[i0 + 1, i1];
							ysidepts[0].Y = yb[i0 + 1, i1];
							zlevel =  - (int)(Z_Mean[i0, i1] * ZScale);
							toppts[0].Y = ysidepts[1].Y = yb[i0 + 1, i1] + zlevel;
							toppts[1].X = ysidepts[2].X = ysidepts[3].X = xb[i0 + 1, i1 + 1];
							ysidepts[3].Y = yb[i0 + 1, i1 + 1];
							toppts[1].Y = ysidepts[2].Y = yb[i0 + 1, i1 + 1] + zlevel;
							gPB.FillPolygon(YSideBrush, ysidepts);
							xsidepts[0].X = xsidepts[1].X = xb[i0 + 1, i1 + 1];
							xsidepts[0].Y = yb[i0 + 1, i1 + 1];
							xsidepts[1].Y = yb[i0 + 1, i1 + 1] + zlevel;
							toppts[2].X = xsidepts[2].X = xsidepts[3].X = xb[i0, i1 + 1];
							xsidepts[3].Y = yb[i0, i1 + 1];
							toppts[2].Y = xsidepts[2].Y = yb[i0, i1 + 1] + zlevel;
							gPB.FillPolygon(XSideBrush, xsidepts);
							toppts[3].X = xb[i0, i1];
							toppts[3].Y = yb[i0, i1] + zlevel;
							gPB.FillPolygon(TopBrush, toppts);
							for (i = 0; i < zticklevels.Length && zticklevels[i] >= zlevel; i++)
							{
								gPB.DrawLine(LevelPen, ysidepts[0].X, ysidepts[0].Y + zticklevels[i], ysidepts[3].X, ysidepts[3].Y + zticklevels[i]);
								gPB.DrawLine(LevelPen, xsidepts[0].X, xsidepts[0].Y + zticklevels[i], xsidepts[3].X, xsidepts[3].Y + zticklevels[i]);
							}
						}							
					gPB.DrawLine(myPen, bottomboxpts[2], topboxpts[2]);
					gPB.DrawLine(myPen, bottomboxpts[3], topboxpts[3]);
					break;

				case 6:
					DrawLEGOTicks(gPB, 0, MinX, MaxX, xb[0, 0], yb[0, 0], XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 1, MinY, MaxY, xb[n0, 0], yb[n0, 0], XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 2, 0, MaxZ, xb[0, 0], yb[0, 0], ZScale, ZScale, -ZTickLength, out zticklevels);
					gPB.DrawLine(myPen, bottomboxpts[2], topboxpts[2]);
					gPB.DrawLine(myPen, bottomboxpts[3], topboxpts[3]);
					for (i1 = n1 - 1; i1 >= 0; i1--)
						for (i0 = 0; i0 < n0; i0++)
						{
							toppts[0].X = ysidepts[0].X = ysidepts[1].X = xb[i0 + 1, i1 + 1];
							ysidepts[0].Y = yb[i0 + 1, i1 + 1];
							zlevel =  - (int)(Z_Mean[i0, i1] * ZScale);
							toppts[0].Y = ysidepts[1].Y = yb[i0 + 1, i1 + 1] + zlevel;
							toppts[1].X = ysidepts[2].X = ysidepts[3].X = xb[i0 + 1, i1];
							ysidepts[3].Y = yb[i0 + 1, i1];
							toppts[1].Y = ysidepts[2].Y = yb[i0 + 1, i1] + zlevel;
							gPB.FillPolygon(YSideBrush, ysidepts);
							xsidepts[0].X = xsidepts[1].X = xb[i0 + 1, i1];
							xsidepts[0].Y = yb[i0 + 1, i1];
							xsidepts[1].Y = yb[i0 + 1, i1] + zlevel;
							toppts[2].X = xsidepts[2].X = xsidepts[3].X = xb[i0, i1];
							xsidepts[3].Y = yb[i0, i1];
							toppts[2].Y = xsidepts[2].Y = yb[i0, i1] + zlevel;
							gPB.FillPolygon(XSideBrush, xsidepts);
							toppts[3].X = xb[i0, i1 + 1];
							toppts[3].Y = yb[i0, i1 + 1] + zlevel;
							gPB.FillPolygon(TopBrush, toppts);
							for (i = 0; i < zticklevels.Length && zticklevels[i] >= zlevel; i++)
							{
								gPB.DrawLine(LevelPen, ysidepts[0].X, ysidepts[0].Y + zticklevels[i], ysidepts[3].X, ysidepts[3].Y + zticklevels[i]);
								gPB.DrawLine(LevelPen, xsidepts[0].X, xsidepts[0].Y + zticklevels[i], xsidepts[3].X, xsidepts[3].Y + zticklevels[i]);
							}
						}							
					gPB.DrawLine(myPen, bottomboxpts[0], topboxpts[0]);
					gPB.DrawLine(myPen, bottomboxpts[1], topboxpts[1]);
					break;

				case 7:
					DrawLEGOTicks(gPB, 0, MinX, MaxX, xb[0, 0], yb[0, 0], XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 1, MinY, MaxY, xb[n0, 0], yb[n0, 0], XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 2, 0, MaxZ, xb[0, 0], yb[0, 0], ZScale, ZScale, -ZTickLength, out zticklevels);
					gPB.DrawLine(myPen, bottomboxpts[2], topboxpts[2]);
					gPB.DrawLine(myPen, bottomboxpts[3], topboxpts[3]);
					for (i0 = 0; i0 < n0; i0++)
						for (i1 = n1 - 1; i1 >= 0; i1--)
						{
							toppts[0].X = ysidepts[0].X = ysidepts[1].X = xb[i0 + 1, i1 + 1];
							ysidepts[0].Y = yb[i0 + 1, i1 + 1];
							zlevel =  - (int)(Z_Mean[i0, i1] * ZScale);
							toppts[0].Y = ysidepts[1].Y = yb[i0 + 1, i1 + 1] + zlevel;
							toppts[1].X = ysidepts[2].X = ysidepts[3].X = xb[i0 + 1, i1];
							ysidepts[3].Y = yb[i0 + 1, i1];
							toppts[1].Y = ysidepts[2].Y = yb[i0 + 1, i1] + zlevel;
							gPB.FillPolygon(YSideBrush, ysidepts);
							xsidepts[0].X = xsidepts[1].X = xb[i0 + 1, i1];
							xsidepts[0].Y = yb[i0 + 1, i1];
							xsidepts[1].Y = yb[i0 + 1, i1] + zlevel;
							toppts[2].X = xsidepts[2].X = xsidepts[3].X = xb[i0, i1];
							xsidepts[3].Y = yb[i0, i1];
							toppts[2].Y = xsidepts[2].Y = yb[i0, i1] + zlevel;
							gPB.FillPolygon(XSideBrush, xsidepts);
							toppts[3].X = xb[i0, i1 + 1];
							toppts[3].Y = yb[i0, i1 + 1] + zlevel;
							gPB.FillPolygon(TopBrush, toppts);
							for (i = 0; i < zticklevels.Length && zticklevels[i] >= zlevel; i++)
							{
								gPB.DrawLine(LevelPen, ysidepts[0].X, ysidepts[0].Y + zticklevels[i], ysidepts[3].X, ysidepts[3].Y + zticklevels[i]);
								gPB.DrawLine(LevelPen, xsidepts[0].X, xsidepts[0].Y + zticklevels[i], xsidepts[3].X, xsidepts[3].Y + zticklevels[i]);
							}
						}							
					gPB.DrawLine(myPen, bottomboxpts[0], topboxpts[0]);
					gPB.DrawLine(myPen, bottomboxpts[1], topboxpts[1]);
					break;

			}
			gPB.DrawPolygon(myPen, topboxpts);

			DrawPanel(gPB, 1);

            return o_vals;
		}

		public static System.Drawing.Color Hue(double h)
		{
            if (h < 0.0) return Color.FromArgb(255, 0, 255);
			if (h < 0.2) return Color.FromArgb((int)(255 * 5.0 * (0.2 - h)), 0, 255);
			if ((h -= 0.2) < 0.2) return Color.FromArgb(0, (int)(255 * 5 * h), 255);
			if ((h -= 0.2) < 0.2) return Color.FromArgb(0, 255, (int)(255 * 5 * (0.2 - h)));
			if ((h -= 0.2) < 0.2) return Color.FromArgb((int)(255 * 5 * h), 255, 0);
            if (h < 1.0) return Color.FromArgb(255, (int)(255 * 5 * (0.2 - (h - 0.2))), 0);
            return Color.FromArgb(255, 255, 0);
		}

		public double [][] Scatter3D(System.Drawing.Graphics gPB, int Width, int Height)
		{
            SetQuality(gPB);
			const int XYTickLength = 8;
			const int ZTickLength = 4;

			double X = 0, Y = 0;
			double MinX = 0, MaxY = 0;
			double MaxX = 0, MinY = 0;
			double MinZ = 0, MaxZ = 0;			
			int i, i0, i1;
			int [] zticklevels = null;

			m_ThreeDPlotC = Math.Cos(m_ObservationAngle / 180.0 * Math.PI);
			m_ThreeDPlotS = Math.Sin(m_ObservationAngle / 180.0 * Math.PI);			

			//gPB.Clear(Color.White);

			if (m_SetXDefaultLimits || (!m_SetXDefaultLimits && m_MaxX<=m_MinX) )
				Fitting.FindStatistics(m_VecX, ref m_MaxX, ref m_MinX, ref X, ref Y);
			if (m_SetYDefaultLimits || (!m_SetYDefaultLimits && m_MaxY<=m_MinY) )
				Fitting.FindStatistics(m_VecY, ref m_MaxY, ref m_MinY, ref X, ref Y);
			Fitting.FindStatistics(m_VecZ, ref m_MaxZ, ref m_MinZ, ref X, ref Y);

			double XRange, YRange, ZRange;
			XRange = (m_MaxX - m_MinX); if (XRange <= 0.0) XRange = Math.Max(Math.Abs(MinX), Math.Abs(MaxX)); if (XRange <= 0.0) XRange = 1.0;
			YRange = (m_MaxY - m_MinY); if (YRange <= 0.0) YRange = Math.Max(Math.Abs(MinY), Math.Abs(MaxY)); if (YRange <= 0.0) YRange = 1.0;
			ZRange = (m_MaxZ - m_MinZ); if (ZRange <= 0.0) ZRange = Math.Max(Math.Abs(MinZ), Math.Abs(MaxZ)); if (ZRange <= 0.0) ZRange = 1.0;
			
			MinX = m_MinX - XRange * 0.05; MaxX = m_MaxX + XRange * 0.05; XRange = MaxX - MinX; m_ThreeDPlotXRescale = 1.0 / XRange;
			MinY = m_MinY - YRange * 0.05; MaxY = m_MaxY + YRange * 0.05; YRange = MaxY - MinY; m_ThreeDPlotYRescale = 1.0 / YRange;
			MinZ = m_MinZ - ZRange * 0.05; MaxZ = m_MaxZ + ZRange * 0.05; ZRange = MaxZ - MinZ;

			double XScale, YScale, ZScale;
			double MinXT, MaxXT, MinYT, MaxYT;

			ThreeDTransform(MinX, MinY, out MinXT, out MinYT);
			MaxXT = MinXT;
			MaxYT = MinYT;
			ThreeDTransform(MaxX, MinY, out X, out Y);
			if (X < MinXT) MinXT = X;
			if (X > MaxXT) MaxXT = X;
			if (Y < MinYT) MinYT = Y;
			if (Y > MaxYT) MaxYT = Y;
			ThreeDTransform(MinX, MaxY, out X, out Y);
			if (X < MinXT) MinXT = X;
			if (X > MaxXT) MaxXT = X;
			if (Y < MinYT) MinYT = Y;
			if (Y > MaxYT) MaxYT = Y;
			ThreeDTransform(MaxX, MaxY, out X, out Y);
			if (X < MinXT) MinXT = X;
			if (X > MaxXT) MaxXT = X;
			if (Y < MinYT) MinYT = Y;
			if (Y > MaxYT) MaxYT = Y;

			int n = m_VecZ.Length;

			int xprec = (int)Math.Ceiling(-Math.Log10(XRange)) + 1; if (xprec < 0) xprec = 0;
			int yprec = (int)Math.Ceiling(-Math.Log10(YRange)) + 1; if (yprec < 0) yprec = 0;
			int zprec = (int)Math.Ceiling(-Math.Log10(ZRange)) + 1; if (zprec < 0) zprec = 0;
			
			int XSizeOfTitles = 2 * (int)Math.Max(
				Math.Max(gPB.MeasureString(MinZ.ToString("F" + zprec), m_LabelFont).Width + Math.Abs(ZTickLength), gPB.MeasureString(MaxZ.ToString("F" + zprec), m_LabelFont).Width + Math.Abs(ZTickLength)), 
				Math.Max(
					Math.Max(gPB.MeasureString(MinX.ToString("F" + xprec), m_LabelFont).Width + Math.Abs(XYTickLength), gPB.MeasureString(MaxX.ToString("F" + xprec), m_LabelFont).Width + Math.Abs(XYTickLength)), 
					Math.Max(gPB.MeasureString(MinY.ToString("F" + xprec), m_LabelFont).Width + Math.Abs(XYTickLength), gPB.MeasureString(MaxY.ToString("F" + yprec), m_LabelFont).Width + Math.Abs(XYTickLength))
				));			

			if (MaxXT > MinXT) XScale = (Width - XSizeOfTitles) / (MaxXT - MinXT);
			else XScale = 1.0;

			YScale = (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) * m_Skewedness / (MaxYT - MinYT);
			ZScale = (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) * (1.0 - m_Skewedness) / ZRange;

			Point [] topboxpts = new Point[4];
			Point [] bottomboxpts = new Point[4];
			Point P = new Point(0, 0);
			ThreeDTransform(MinX, MinY, out X, out Y);
			bottomboxpts[0].X = topboxpts[0].X = (int)((X - MinXT) * XScale) + XSizeOfTitles / 2;
			bottomboxpts[0].Y = m_LabelFont.Height + (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) - (int)((Y - MinYT) * YScale) + 1;
			ThreeDTransform(MaxX, MinY, out X, out Y);
			bottomboxpts[1].X = topboxpts[1].X = (int)((X - MinXT) * XScale) + XSizeOfTitles / 2;
			bottomboxpts[1].Y = m_LabelFont.Height + (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) - (int)((Y - MinYT) * YScale) + 1;
			ThreeDTransform(MaxX, MaxY, out X, out Y);
			bottomboxpts[2].X = topboxpts[2].X = (int)((X - MinXT) * XScale) + XSizeOfTitles / 2;
			bottomboxpts[2].Y = m_LabelFont.Height + (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) - (int)((Y - MinYT) * YScale) + 1;
			ThreeDTransform(MinX, MaxY, out X, out Y);
			bottomboxpts[3].X = topboxpts[3].X = (int)((X - MinXT) * XScale) + XSizeOfTitles / 2;
			bottomboxpts[3].Y = m_LabelFont.Height + (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) - (int)((Y - MinYT) * YScale) + 1;
			gPB.DrawPolygon(myPen, bottomboxpts);
			topboxpts[0].Y = bottomboxpts[0].Y - (int)(ZRange * ZScale) - 1;
			topboxpts[1].Y = bottomboxpts[1].Y - (int)(ZRange * ZScale) - 1;
			topboxpts[2].Y = bottomboxpts[2].Y - (int)(ZRange * ZScale) - 1;
			topboxpts[3].Y = bottomboxpts[3].Y - (int)(ZRange * ZScale) - 1;
			switch ((int)(m_ObservationAngle / 45.0))
			{
				case 0:
					DrawLEGOTicks(gPB, 0, MinX, MaxX, bottomboxpts[0].X, bottomboxpts[0].Y, XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 1, MinY, MaxY, bottomboxpts[0].X, bottomboxpts[0].Y, XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 3, MinZ, MaxZ, bottomboxpts[3].X, bottomboxpts[3].Y, ZScale, ZScale, -ZTickLength, out zticklevels);
					gPB.DrawLine(myPen, bottomboxpts[2], topboxpts[2]);
					gPB.DrawLine(myPen, bottomboxpts[3], topboxpts[3]);
					for (i = 0; i < n; i++)
					{
						double zlev = m_VecZ[i] - m_MinZ;
						ThreeDTransform(m_VecX[i], m_VecY[i], out X, out Y);
						P.X = (int)((X - MinXT) * XScale) + XSizeOfTitles / 2;
						P.Y = m_LabelFont.Height + (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) - (int)((Y - MinYT) * YScale) + 1 - (int)(ZScale * zlev);
						zlev /= ZRange;
                        m_Marker.Draw(gPB, new Pen(Hue(zlev)), new SolidBrush(Hue(zlev)), m_MarkerSize, P.X, P.Y);
					}
					gPB.DrawLine(myPen, bottomboxpts[0], topboxpts[0]);
					gPB.DrawLine(myPen, bottomboxpts[1], topboxpts[1]);
					break;

				case 1:
					DrawLEGOTicks(gPB, 0, MinX, MaxX, bottomboxpts[0].X, bottomboxpts[0].Y, XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 1, MinY, MaxY, bottomboxpts[0].X, bottomboxpts[0].Y, XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 3, MinZ, MaxZ, bottomboxpts[3].X, bottomboxpts[3].Y, ZScale, ZScale, -ZTickLength, out zticklevels);
					gPB.DrawLine(myPen, bottomboxpts[2], topboxpts[2]);
					gPB.DrawLine(myPen, bottomboxpts[3], topboxpts[3]);
					for (i = 0; i < n; i++)
					{
						double zlev = m_VecZ[i] - m_MinZ;
						ThreeDTransform(m_VecX[i], m_VecY[i], out X, out Y);
						P.X = (int)((X - MinXT) * XScale) + XSizeOfTitles / 2;
						P.Y = m_LabelFont.Height + (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) - (int)((Y - MinYT) * YScale) + 1 - (int)(ZScale * zlev);
						zlev /= ZRange;
						gPB.FillEllipse(new SolidBrush(Hue(zlev)), P.X - 1, P.Y - 1, 3, 3);
					}
					gPB.DrawLine(myPen, bottomboxpts[0], topboxpts[0]);
					gPB.DrawLine(myPen, bottomboxpts[1], topboxpts[1]);
					break;

				case 2:
					DrawLEGOTicks(gPB, 0, MinX, MaxX, bottomboxpts[3].X, bottomboxpts[3].Y, XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 1, MinY, MaxY, bottomboxpts[0].X, bottomboxpts[0].Y, XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 3, MinZ, MaxZ, bottomboxpts[2].X, bottomboxpts[2].Y, ZScale, ZScale, -ZTickLength, out zticklevels);
					gPB.DrawLine(myPen, bottomboxpts[0], topboxpts[0]);
					gPB.DrawLine(myPen, bottomboxpts[1], topboxpts[1]);
					for (i = 0; i < n; i++)
					{
						double zlev = m_VecZ[i] - m_MinZ;
						ThreeDTransform(m_VecX[i], m_VecY[i], out X, out Y);
						P.X = (int)((X - MinXT) * XScale) + XSizeOfTitles / 2;
						P.Y = m_LabelFont.Height + (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) - (int)((Y - MinYT) * YScale) + 1 - (int)(ZScale * zlev);
						zlev /= ZRange;
						gPB.FillEllipse(new SolidBrush(Hue(zlev)), P.X - 1, P.Y - 1, 3, 3);
					}
					gPB.DrawLine(myPen, bottomboxpts[2], topboxpts[2]);
					gPB.DrawLine(myPen, bottomboxpts[3], topboxpts[3]);
					break;

				case 3:
					DrawLEGOTicks(gPB, 0, MinX, MaxX, bottomboxpts[3].X, bottomboxpts[3].Y, XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 1, MinY, MaxY, bottomboxpts[0].X, bottomboxpts[0].Y, XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 3, MinZ, MaxZ, bottomboxpts[2].X, bottomboxpts[2].Y, ZScale, ZScale, -ZTickLength, out zticklevels);
					gPB.DrawLine(myPen, bottomboxpts[0], topboxpts[0]);
					gPB.DrawLine(myPen, bottomboxpts[1], topboxpts[1]);
					for (i = 0; i < n; i++)
					{
						double zlev = m_VecZ[i] - m_MinZ;
						ThreeDTransform(m_VecX[i], m_VecY[i], out X, out Y);
						P.X = (int)((X - MinXT) * XScale) + XSizeOfTitles / 2;
						P.Y = m_LabelFont.Height + (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) - (int)((Y - MinYT) * YScale) + 1 - (int)(ZScale * zlev);
						zlev /= ZRange;
                        
						gPB.FillEllipse(new SolidBrush(Hue(zlev)), P.X - 1, P.Y - 1, 3, 3);
					}
					gPB.DrawLine(myPen, bottomboxpts[2], topboxpts[2]);
					gPB.DrawLine(myPen, bottomboxpts[3], topboxpts[3]);
					break;

				case 4:
					DrawLEGOTicks(gPB, 0, MinX, MaxX, bottomboxpts[3].X, bottomboxpts[3].Y, XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 1, MinY, MaxY, bottomboxpts[1].X, bottomboxpts[1].Y, XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 3, MinZ, MaxZ, bottomboxpts[1].X, bottomboxpts[1].Y, ZScale, ZScale, -ZTickLength, out zticklevels);
					gPB.DrawLine(myPen, bottomboxpts[0], topboxpts[0]);
					gPB.DrawLine(myPen, bottomboxpts[1], topboxpts[1]);
					for (i = 0; i < n; i++)
					{
						double zlev = m_VecZ[i] - m_MinZ;
						ThreeDTransform(m_VecX[i], m_VecY[i], out X, out Y);
						P.X = (int)((X - MinXT) * XScale) + XSizeOfTitles / 2;
						P.Y = m_LabelFont.Height + (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) - (int)((Y - MinYT) * YScale) + 1 - (int)(ZScale * zlev);
						zlev /= ZRange;
						gPB.FillEllipse(new SolidBrush(Hue(zlev)), P.X - 1, P.Y - 1, 3, 3);
					}
					gPB.DrawLine(myPen, bottomboxpts[2], topboxpts[2]);
					gPB.DrawLine(myPen, bottomboxpts[3], topboxpts[3]);
					break;

				case 5:
					DrawLEGOTicks(gPB, 0, MinX, MaxX, bottomboxpts[3].X, bottomboxpts[3].Y, XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 1, MinY, MaxY, bottomboxpts[1].X, bottomboxpts[1].Y, XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 3, MinZ, MaxZ, bottomboxpts[1].X, bottomboxpts[1].Y, ZScale, ZScale, -ZTickLength, out zticklevels);
					gPB.DrawLine(myPen, bottomboxpts[0], topboxpts[0]);
					gPB.DrawLine(myPen, bottomboxpts[1], topboxpts[1]);
					for (i = 0; i < n; i++)
					{
						double zlev = m_VecZ[i] - m_MinZ;
						ThreeDTransform(m_VecX[i], m_VecY[i], out X, out Y);
						P.X = (int)((X - MinXT) * XScale) + XSizeOfTitles / 2;
						P.Y = m_LabelFont.Height + (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) - (int)((Y - MinYT) * YScale) + 1 - (int)(ZScale * zlev);
						zlev /= ZRange;
						gPB.FillEllipse(new SolidBrush(Hue(zlev)), P.X - 1, P.Y - 1, 3, 3);
					}
					gPB.DrawLine(myPen, bottomboxpts[2], topboxpts[2]);
					gPB.DrawLine(myPen, bottomboxpts[3], topboxpts[3]);
					break;

				case 6:
					DrawLEGOTicks(gPB, 0, MinX, MaxX, bottomboxpts[0].X, bottomboxpts[0].Y, XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 1, MinY, MaxY, bottomboxpts[1].X, bottomboxpts[1].Y, XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 3, MinZ, MaxZ, bottomboxpts[0].X, bottomboxpts[0].Y, ZScale, ZScale, -ZTickLength, out zticklevels);
					gPB.DrawLine(myPen, bottomboxpts[2], topboxpts[2]);
					gPB.DrawLine(myPen, bottomboxpts[3], topboxpts[3]);
					for (i = 0; i < n; i++)
					{
						double zlev = m_VecZ[i] - m_MinZ;
						ThreeDTransform(m_VecX[i], m_VecY[i], out X, out Y);
						P.X = (int)((X - MinXT) * XScale) + XSizeOfTitles / 2;
						P.Y = m_LabelFont.Height + (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) - (int)((Y - MinYT) * YScale) + 1 - (int)(ZScale * zlev);
						zlev /= ZRange;
						gPB.FillEllipse(new SolidBrush(Hue(zlev)), P.X - 1, P.Y - 1, 3, 3);
					}
					gPB.DrawLine(myPen, bottomboxpts[0], topboxpts[0]);
					gPB.DrawLine(myPen, bottomboxpts[1], topboxpts[1]);
					break;

				case 7:
					DrawLEGOTicks(gPB, 0, MinX, MaxX, bottomboxpts[0].X, bottomboxpts[0].Y, XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 1, MinY, MaxY, bottomboxpts[1].X, bottomboxpts[1].Y, XScale, YScale, XYTickLength, out zticklevels);
					DrawLEGOTicks(gPB, 3, MinZ, MaxZ, bottomboxpts[0].X, bottomboxpts[0].Y, ZScale, ZScale, -ZTickLength, out zticklevels);
					gPB.DrawLine(myPen, bottomboxpts[2], topboxpts[2]);
					gPB.DrawLine(myPen, bottomboxpts[3], topboxpts[3]);
					for (i = 0; i < n; i++)
					{
						double zlev = m_VecZ[i] - m_MinZ;
						ThreeDTransform(m_VecX[i], m_VecY[i], out X, out Y);
						P.X = (int)((X - MinXT) * XScale) + XSizeOfTitles / 2;
						P.Y = m_LabelFont.Height + (Height - 2 * m_LabelFont.Height - 2 * Math.Abs(XYTickLength)) - (int)((Y - MinYT) * YScale) + 1 - (int)(ZScale * zlev);
						zlev /= ZRange;
						gPB.FillEllipse(new SolidBrush(Hue(zlev)), P.X - 1, P.Y - 1, 3, 3);
					}
					gPB.DrawLine(myPen, bottomboxpts[0], topboxpts[0]);
					gPB.DrawLine(myPen, bottomboxpts[1], topboxpts[1]);
					break;

			}
			gPB.DrawPolygon(myPen, topboxpts);

			DrawPanel(gPB, 3);

            return null;
		}


		private struct TickMark
		{
			public System.Drawing.Rectangle Rect;
			public string Text;

			public TickMark(int x, int y, int w, int h, string text)
			{
				Rect = new System.Drawing.Rectangle(x - w / 2, y - h / 2, w, h);
				Text = text;
			}
		}

		private string m_Panel = null;

		public string Panel
		{
			get { return (m_Panel == null) ? null : (string)m_Panel.Clone(); }
			set { m_Panel = value; }
		}

		private string m_PanelFormat = null;

		public string PanelFormat
		{
			get { return (m_PanelFormat == null) ? null : (string)m_PanelFormat.Clone(); }
			set { m_PanelFormat = value; }
		}

		private Font m_PanelFont = new Font("Comic Sans MS", 9, System.Drawing.FontStyle.Bold);

		public Font PanelFont
		{
			get { return (m_PanelFont == null) ? null : (Font)m_PanelFont.Clone(); }
			set 
			{ 
				m_PanelFont = value;
 				if (m_PanelFont == null)
					m_PanelFont = new Font("Comic Sans MS", 9, System.Drawing.FontStyle.Bold);
			}
		}

		private double m_PanelX = 1.0;

		public double PanelX
		{
			get { return m_PanelX; }
			set { m_PanelX = value; }
		}

		private double m_PanelY = 0.0;

		public double PanelY
		{
			get { return m_PanelY; }
			set { m_PanelY = value; }
		}

		private void DrawPanel(System.Drawing.Graphics g, int vars)
		{
			string panel;
			if (m_Panel == null) 
			{
				int i;
				if (vars == 1) panel = m_XTitle + " counts\r\n";
				else if (vars == 2) panel = m_YTitle + " vs. " + m_XTitle + "\r\n";
				else if (vars == 3) panel = m_ZTitle + " vs. " + m_XTitle + " and " + m_YTitle + "\r\n";
				else panel = "";
				panel += "Entries: " + this.m_VecX.Length;
				if (m_ParDescr != null)
					for (i = 0; i < m_ParDescr.Length; i++)
						if (m_ParDescr[i] != null && m_ParDescr[i].Trim().Length > 0)
							panel += "\r \n" + m_ParDescr[i] + ": " + ((m_PanelFormat == null) ?
								FitPar[i].ToString(System.Globalization.CultureInfo.InvariantCulture) :
								FitPar[i].ToString(m_PanelFormat, System.Globalization.CultureInfo.InvariantCulture)
								);                
			}
			else if (m_Panel.Trim().Length == 0) return;
			else panel = m_Panel;			
			System.Drawing.SizeF size = g.MeasureString(panel, m_PanelFont);
			size.Width += 4;
			size.Height += 4;
			float x, y;
			x = (float)((g.VisibleClipBounds.Width - size.Width - 1.0) * m_PanelX + g.VisibleClipBounds.Left);
			y = (float)((g.VisibleClipBounds.Height - size.Height - 1.0) * m_PanelY + g.VisibleClipBounds.Top);
			g.FillRectangle(new System.Drawing.SolidBrush(System.Drawing.Color.White), x, y, size.Width, size.Height);
			g.DrawRectangle(new System.Drawing.Pen(System.Drawing.Color.Black, 1), x, y, size.Width, size.Height);
			g.DrawString(panel, m_PanelFont, new System.Drawing.SolidBrush(System.Drawing.Color.Black), x + 2.0f, y + 2.0f);
		}

		private void DrawLEGOTicks(System.Drawing.Graphics g, int axis, double min, double max, int xstart, int ystart, double xscale, double yscale, int ticklen, out int [] zticklevels)
		{			
			double tickstep, mintick;
			double range = max - min;
			int divfactor = 0;
			int steps, s;			
			if (max <= min) max = min + 1.0;			
			tickstep = Math.Pow(10.0, Math.Ceiling(Math.Log10(max - min)));
			zticklevels = null;
			int DTickX, DTickY;
			while (tickstep >= range)
			{
				divfactor = (++divfactor % 3);
				switch (divfactor)
				{
					case 1:		tickstep *= 0.5; break;
					case 2:		tickstep *= 0.4; break;
					case 0:		tickstep *= 0.5; break;
				}
			}
			TickMark [] Ticks = null, OldTicks = null;
			double X = 0, Y = 0, v;
			int prec;
			string label;

			switch (axis)
			{
				// X
				case 0:
				// Y
				case 1:
							do
							{
								OldTicks = Ticks;
								mintick = Math.Ceiling(min / tickstep) * tickstep;
								prec = (int)Math.Ceiling(-Math.Log10(tickstep)); if (prec < 0) prec = 0;
								steps = (int)Math.Ceiling((max - mintick) / tickstep);
								Ticks = new TickMark[steps];
								if (axis == 0) ThreeDTransform(0, 1, out X, out Y);
								else if (axis == 1) ThreeDTransform(1, 0, out X, out Y);
								X *= xscale;
								Y *= -yscale;
								if (Y < 0)
								{
									X = -X;
									Y = -Y;
								}
								v = Math.Abs(ticklen) / Math.Max(Math.Abs(X), Math.Abs(Y));
								DTickX = (int)(v * X);
								DTickY = (int)(v * Y);
								for (s = 0; s < steps; s++)
								{				
									v = mintick + s * tickstep;
									if (axis == 0) ThreeDTransform(v - min, 0, out X, out Y);
									else ThreeDTransform(0, v - min, out X, out Y);
									label = v.ToString("F" + prec);
									System.Drawing.SizeF size = g.MeasureString(label, m_LabelFont);
									Ticks[s] = new TickMark(xstart + (int)(X * xscale), 
										ystart - (int)(Y * yscale), (int)size.Width + 2, (int)size.Height + 2, label);
									if (s > 0 && OldTicks != null)
									{
										if (Ticks[s].Rect.IntersectsWith(Ticks[s - 1].Rect))
										{
											Ticks = OldTicks;
											break;
										}
									}
								}
								divfactor = (++divfactor % 3);
								switch (divfactor)
								{
									case 1:		tickstep *= 0.5; break;
									case 2:		tickstep *= 0.4; break;
									case 0:		tickstep *= 0.5; break;
								}
							}
							while (Ticks != OldTicks);
							steps = Ticks.Length;
							for (s = 0; s < steps; s++)							
							{								
								int tx, ty;
								g.DrawLine(myPen, (Ticks[s].Rect.Left + Ticks[s].Rect.Right) / 2, (Ticks[s].Rect.Top + Ticks[s].Rect.Bottom) / 2,
									tx = (Ticks[s].Rect.Left + Ticks[s].Rect.Right) / 2 + DTickX, ty = (Ticks[s].Rect.Top + Ticks[s].Rect.Bottom) / 2 + DTickY);
								Ticks[s].Rect.Offset(tx - Ticks[s].Rect.Left - ((DTickX < 0) ? Ticks[s].Rect.Width : 0), ty - Ticks[s].Rect.Top);
								g.DrawString(Ticks[s].Text, m_LabelFont, myBrush, Ticks[s].Rect);
							}
							TickMark tc = Ticks[steps / 2];
							tc.Rect.Offset(0, m_LabelFont.Height);
							tc.Text = (axis == 0) ? m_XTitle : m_YTitle;
							tc.Rect.Inflate(((int)g.MeasureString(tc.Text, m_LabelFont).Width + 2 - tc.Rect.Width) / 2, 0);
							g.DrawString(tc.Text, m_LabelFont, myBrush, tc.Rect);							
							break;

				// Z
				case 2: 
				// Z with label
				case 3:							
							do
							{
								OldTicks = Ticks;
								mintick = Math.Ceiling(min / tickstep) * tickstep;
								prec = (int)Math.Ceiling(-Math.Log10(tickstep)); if (prec < 0) prec = 0;
								steps = (int)((max - mintick) / tickstep) + 1;
								if (axis == 2)
								{
									Ticks = new TickMark[steps - 1];
									for (s = 1; s < steps; s++)
									{				
										v = mintick + s * tickstep;
										label = v.ToString("F" + prec);
										System.Drawing.SizeF size = g.MeasureString(label, m_LabelFont);
										Ticks[s - 1] = new TickMark(xstart + ticklen + Math.Sign(ticklen) * ((int)size.Width + 2) / 2, ystart - (int)((v - min) * xscale /* it is actually zscale */), (int)size.Width + 2, (int)size.Height + 2, label);
										if (s > 1 && OldTicks != null)
										{
											if (Ticks[s - 1].Rect.IntersectsWith(Ticks[s - 2].Rect))
											{
												Ticks = OldTicks;
												break;
											}
										}
									}
									divfactor = (++divfactor % 3);
									switch (divfactor)
									{
										case 1:		tickstep *= 0.5; break;
										case 2:		tickstep *= 0.4; break;
										case 0:		tickstep *= 0.5; break;
									}
								}
								else
								{
									Ticks = new TickMark[steps];
									for (s = 0; s < steps; s++)
									{				
										v = mintick + s * tickstep;
										label = v.ToString("F" + prec);
										System.Drawing.SizeF size = g.MeasureString(label, m_LabelFont);
										Ticks[s] = new TickMark(xstart + ticklen + Math.Sign(ticklen) * ((int)size.Width + 2) / 2, ystart - (int)((v - min) * xscale /* it is actually zscale */), (int)size.Width + 2, (int)size.Height + 2, label);
										if (s > 0 && OldTicks != null)
										{
											if (Ticks[s].Rect.IntersectsWith(Ticks[s - 1].Rect))
											{
												Ticks = OldTicks;
												break;
											}
										}
									}
									divfactor = (++divfactor % 3);
									switch (divfactor)
									{
										case 1:		tickstep *= 0.5; break;
										case 2:		tickstep *= 0.4; break;
										case 0:		tickstep *= 0.5; break;
									}
								}
							}
							while (Ticks != OldTicks);
							steps = Ticks.Length;
							zticklevels = new int[steps];
							for (s = 0; s < steps; s++)
							{
								zticklevels[s] = (Ticks[s].Rect.Top + Ticks[s].Rect.Bottom) / 2 - ystart;
								g.DrawLine(myPen, xstart, ystart + zticklevels[s], xstart + ticklen, ystart + zticklevels[s]);
							}
							foreach (TickMark t in Ticks)
								g.DrawString(t.Text, m_LabelFont, myBrush, t.Rect);
							tc = Ticks[steps - 1];
							tc.Rect.Offset(0, -m_LabelFont.Height);
							tc.Text = (axis == 3) ? m_ZTitle : "Counts";
							int dx = ((int)g.MeasureString(tc.Text, m_LabelFont).Width + 2 - tc.Rect.Width) / 2;
							tc.Rect.Inflate(dx, 0);
							tc.Rect.Offset(dx * Math.Sign(-ticklen), 0);
							g.DrawString(tc.Text, m_LabelFont, myBrush, tc.Rect);
							break;
			}
		}

		public double [][] GAreaComputedValues(System.Drawing.Graphics gPB, int Width, int Height)
		{
            SetQuality(gPB);
			double MinX =0, MaxY =0, MedX =0, RMSx =0;
			double MaxX =0, MinY =0, MedY =0, RMSy =0;
			double MaxZ =0, MinZ =0;
			double[] X_Mean, Y_Mean; 
			double [,] Z_Mean, rmsZ_Mean;
			int [,] nEnt;
			int i=0, j;


			//gPB.Clear(Color.White);

			Fitting.Prepare_2DCustom_Distribution_ZVal(m_VecX, m_VecY, m_VecZ, m_DX, m_DY, 
				out X_Mean,out Y_Mean, 
				out Z_Mean, out rmsZ_Mean, out nEnt);

			if (m_SetXDefaultLimits || (!m_SetXDefaultLimits && m_MaxX<=m_MinX) )
				Fitting.FindStatistics(X_Mean, ref m_MaxX, ref m_MinX, ref MedX, ref RMSx);
			if (m_SetYDefaultLimits || (!m_SetYDefaultLimits && m_MaxY<=m_MinY) )
				Fitting.FindStatistics(Y_Mean, ref m_MaxY, ref m_MinY, ref MedY, ref RMSy);

			//Pulisce i vecchi parametri
			m_FitPar = new double [1];
			m_ParDescr= new string [1];

			MinX=m_MinX; MaxX=m_MaxX; 
			MinY=m_MinY; MaxY=m_MaxY;

			MinX = MinX - m_DX/2;
			MaxX = MaxX + m_DX/2;
			MinY = MinY - m_DY/2;
			MaxY = MaxY + m_DY/2;

			float AggX=0, AggY=0, LengthX=0, LengthY=0;
			float AggFontX=0, AggFontY=0;
			string StrY="";

			SetBorders(gPB, MaxX, MinX, MaxY, MinY, Height, Width, 
				ref AggX, ref AggY, ref LengthX, ref LengthY,
				ref AggFontX, ref AggFontY, ref StrY);

			int n = Z_Mean.GetLength(0);
			int m = Z_Mean.GetLength(1);
			MaxZ = Z_Mean[0,0];
			MinZ = MaxZ;

            double[][] o_vals = new double[4][] { new double[n * m], new double[n * m], new double[n * m], new double[n * m] };

			for(i=0;i<n;i++)
				for(j=0;j<m;j++)
				{
                    o_vals[0][i * m + j] = X_Mean[i];
                    o_vals[1][i * m + j] = Y_Mean[j];
                    o_vals[2][i * m + j] = Z_Mean[i, j];
                    o_vals[3][i * m + j] = nEnt[i, j];
                    if (Z_Mean[i, j] > MaxZ) MaxZ = Z_Mean[i, j];
					if (Z_Mean[i,j]<MinZ) MinZ=Z_Mean[i,j];
				};

			byte c;
			for(i=0;i<n;i++)
				for(j=0;j<m;j++)
				{	
					if (((X_Mean[i]-m_DX/2)>= MinX) && ((X_Mean[i]+m_DX/2)<= MaxX))
					{
						c = (byte)(255-(255*(Z_Mean[i,j]-MinZ)/(MaxZ-MinZ)));
						Brush tmpBrush = new SolidBrush(Color.FromArgb(c,c,c));
						gPB.FillRectangle(tmpBrush, AffineX(X_Mean[i] - m_DX/2), AffineY(Y_Mean[j] + m_DY/2), ShrinkX(m_DX), ShrinkY(m_DY)); 
					};
				};

			//Passaggio all'esterno
			m_MaxZ = MaxZ;
			m_MinZ = MinZ;

			System.Drawing.PointF myPt = new System.Drawing.PointF();
			myPt.X = 0;
			myPt.Y = 0;
			gPB.DrawString(m_YTitle, m_LabelFont, myBrush, myPt);

			myPt.X = Width - gPB.MeasureString(m_XTitle,m_LabelFont).Width;
			myPt.Y = Height -gPB.MeasureString(m_XTitle,m_LabelFont).Height;
			gPB.DrawString(m_XTitle, m_LabelFont, myBrush, myPt);

			DrawXAxis(X_Mean.GetLength(0), MinY, MaxX, MinX, MaxY, MinY, AggX, AggY, AggFontX, AggFontY, gPB);
			DrawYAxis(MinX, /*MaxX, MinX,*/ MaxY, MinY, AggX, AggY, AggFontX, AggFontY, StrY, gPB);

			if (m_FunctionOverlayed) OverlayFunction(MaxX, MinX, MaxY, MinY, gPB);

			m_PlottedX = (double[])X_Mean.Clone();
			m_PlottedY = (double[])Y_Mean.Clone();
			m_PlottedMatZ = (double[,])Z_Mean.Clone();

            return o_vals;
		}


		public double [][] Pie(System.Drawing.Graphics gPB, int Width, int Height)
		{
            SetQuality(gPB);
			double MinX =0,  MedX =0, RMSx =0;
			double MaxX =0;
			int i=0, n;
			double Norm;
			string[] Perc;
			double radius, circonf;
			//gPB.Clear(Color.White);

			Fitting.FindStatistics(m_VecX, ref MaxX, ref MinX, ref MedX, ref RMSx);

			//Pulisce i vecchi parametri
			m_FitPar = new double [1];
			m_ParDescr= new string [1];

			n=m_VecX.GetLength(0);
			Perc = new string[n];
			Norm=MedX*n;
			for(i=0;i<n; i++)
			{
				double tmp = m_VecX[i]/Norm;
				Perc[i]=tmp.ToString("F2")+"%";
			};

			radius = (Width/4);
			if (Height/4 < radius) radius =(Height/4);
			circonf = 2*radius*Math.PI;

			byte c;
			float tmpAng=0;
			for(i=0;i<n;i++)
			{
				c = (byte)(255*i/n);
				Brush tmpBrush;
				if(i%6==0)
					tmpBrush = new SolidBrush(Color.FromArgb(255,c,c));
				else if (i%6==1)
					tmpBrush = new SolidBrush(Color.FromArgb(c,255,c));
				else if (i%6==2)
					tmpBrush = new SolidBrush(Color.FromArgb(c,c,255));
				else if (i%6==3)
					tmpBrush = new SolidBrush(Color.FromArgb(255,c,255));
				else if (i%6==4)
					tmpBrush = new SolidBrush(Color.FromArgb(c,255,255));
				else 
					tmpBrush = new SolidBrush(Color.FromArgb(255,255,c));

				gPB.FillPie(tmpBrush, (int)(Width/4), (int)(Height/4),(int)(2*radius),(int)(2*radius), tmpAng, (float)(360*m_VecX[i]/Norm));

				System.Drawing.PointF myPt = new System.Drawing.PointF();
				string tmpString =(m_VecX[i]*100/Norm).ToString("F2")+"% "+ m_CommentX[i];
				if ((m_VecX[i]*100/Norm)<0.01) tmpString =(m_VecX[i]*100/Norm).ToString("F3")+"% "+ m_CommentX[i];
				myPt.X = (float)((Width/4)+ radius*(1+Math.Cos(Math.PI*(tmpAng/180 + m_VecX[i]/Norm))));
				if (Math.Cos(Math.PI*(tmpAng/180 + m_VecX[i]/Norm))<0) myPt.X = myPt.X-gPB.MeasureString(tmpString,m_LabelFont).Width;
				myPt.Y = (float)((Height/4)+ radius*(1+Math.Sin(Math.PI*(tmpAng/180 + m_VecX[i]/Norm))));
				if (Math.Sin(Math.PI*(tmpAng/180 + m_VecX[i]/Norm))<0) myPt.Y = myPt.Y-gPB.MeasureString(tmpString,m_LabelFont).Height;
				gPB.DrawString(tmpString, m_LabelFont, myBrush, myPt);

				tmpAng+=(float)(360*m_VecX[i]/Norm);
			};

            return null;
		}

		#endregion

		#region External Properties

		private double[] m_PlottedX;

		public double[] PlottedX
		{
			get
			{
				return (double[])m_PlottedX.Clone();	
			}

		}

		private double[] m_PlottedY;

		public double[] PlottedY
		{
			get
			{
				return (double[])m_PlottedY.Clone();	
			}

		}


		private double[] m_PlottedSY;

		public double[] PlottedSY
		{
			get
			{
				return (double[])m_PlottedSY.Clone();	
			}

		}

		private double[,] m_PlottedMatZ;

		public double[,] PlottedMatZ
		{
			get
			{
				return (double[,])m_PlottedMatZ.Clone();	
			}

		}

		private string[] m_ParDescr;

		public string[] ParDescr
		{
			get
			{
				return (string[])m_ParDescr.Clone();	
			}

		}

		private double[] m_FitPar = null;

		public double[] FitPar
		{
			get
			{
				if (m_FitPar != null) 
					return (double[])m_FitPar.Clone();	
				else
					return null;
			}

		}

		private string m_Function;

		public string Function
		{
			set
			{
				m_Function = (string)value.Clone();	
			}

		}

		private bool m_FunctionOverlayed;

		public bool FunctionOverlayed
		{
			get
			{
				return m_FunctionOverlayed;
			}

			set
			{
				m_FunctionOverlayed = value;	
			}

		}

		private string [] m_CommentX;

		public string[] CommentX
		{
			set
			{
				m_CommentX = (string[])value.Clone();	
			}

		}

		private double [] m_VecX;
		 
		public double[] VecX
		{
			set
			{
				m_VecX = (double[])value.Clone();	
			}

		}

		private double [] m_VecY;

		public double[] VecY
		{
			set
			{
				m_VecY = (double[])value.Clone();	
			}

		}

		private double [,] m_MatZ;

		public double[,] MatZ
		{
			set
			{
				m_MatZ = (double[,])value.Clone();	
			}

		}

		private double [] m_VecZ;

		public double[] VecZ
		{
			set
			{
				m_VecZ = (double[])value.Clone();	
			}

		}

        private double[] m_VecDX;

        public double[] VecDX
        {
            set
            {
                m_VecDX = (double[])value.Clone();
            }
        }

        private double[] m_VecDY;

        public double[] VecDY
        {
            set
            {
                m_VecDY = (double[])value.Clone();
            }
        }

        private bool m_FittingOnlyDataInPlot = true;

		public bool FittingOnlyDataInPlot
		{
			get
			{
				return m_FittingOnlyDataInPlot;
			}

			set
			{
				m_FittingOnlyDataInPlot = value;	
			}

		}

		private bool m_SetXDefaultLimits=true;

		public bool SetXDefaultLimits
		{
			get
			{
				return m_SetXDefaultLimits;
			}

			set
			{
				m_SetXDefaultLimits = value;	
			}

		}

		private bool m_SetYDefaultLimits=true;

		public bool SetYDefaultLimits
		{
			get
			{
				return m_SetYDefaultLimits;
			}

			set
			{
				m_SetYDefaultLimits = value;	
			}

		}

		private double m_MaxX;

		public double MaxX
		{
			get
			{
				return m_MaxX;
			}

			set
			{
				m_MaxX = value;	
			}

		}

		private double m_MinX;

		public double MinX
		{
			get
			{
				return m_MinX;
			}

			set
			{
				m_MinX = value;	
			}

		}
		
		private float m_DX;

		public float DX
		{
			get
			{
				return m_DX;
			}

			set
			{
				m_DX = value;	
			}

		}

		private string m_XTitle;

		public string XTitle
		{
			get
			{
				return m_XTitle;
			}

			set
			{
				m_XTitle = value;	
			}

		}

		private string m_YTitle;

		public string YTitle
		{
			get
			{
				return m_YTitle;
			}

			set
			{
				m_YTitle = value;	
			}

		}

		private string m_ZTitle;

		public string ZTitle
		{
			get
			{
				return m_ZTitle;
			}

			set
			{
				m_ZTitle = value;	
			}

		}

		private double m_MaxY;

		public double MaxY
		{
			get
			{
				return m_MaxY;
			}

			set
			{
				m_MaxY = value;	
			}

		}
		private double m_MinY;

		public double MinY
		{
			get
			{
				return m_MinY;
			}

			set
			{
				m_MinY = value;	
			}

		}

		private double m_MaxZ;

		public double MaxZ
		{
			get
			{
				return m_MaxZ;
			}

			set
			{
				m_MaxZ = value;	
			}

		}

		private double m_MinZ;

		public double MinZ
		{
			get
			{
				return m_MinZ;
			}

			set
			{
				m_MinZ = value;	
			}

		}

		private float m_DY;

		public float DY
		{
			get
			{
				return m_DY;
			}

			set
			{
				m_DY = value;	
			}

		}

		private short m_HistoFit;

		public short HistoFit
		{
			get
			{
				return m_HistoFit;
			}

			set
			{
				m_HistoFit = value;	
			}

		}

		private short m_ScatterFit;

		public short ScatterFit
		{
			get
			{
				return m_ScatterFit;
			}

			set
			{
				m_ScatterFit = value;	
			}

		}

		private bool m_LinearFitWE;

		public bool LinearFitWE
		{
			get
			{
				return m_LinearFitWE;
			}

			set
			{
				m_LinearFitWE = value;	
			}

		}

		private bool m_HistoFill;

		public bool HistoFill
		{
			get
			{
				return m_HistoFill;
			}

			set
			{
				m_HistoFill = value;	
			}

		}

		private int m_PlotThickness;

		public int PlotThickness
		{
			get 
			{ 
				return m_PlotThickness; 
			}
			set 
			{
				m_PlotThickness = value;
				myRedPen.Dispose();
				myRedPen = new Pen(m_HistoColor, m_PlotThickness);
                myPen.Dispose();
                myPen = new Pen(Color.Black, m_PlotThickness);
			}
		}

		private Color m_HistoColor;

		public Color HistoColor
		{
			get
			{
				return m_HistoColor;
			}

			set
			{
				m_HistoColor = value;	
				myRedPen.Dispose();
				myRedPen = new Pen(m_HistoColor, m_PlotThickness);
			}

		}
		

		#endregion

		#region Internal Parameters
		
		//private Graphics gPB;

	
		
		private float m_alphaX;

		private float alphaX
		{
			get
			{
				return m_alphaX;
			}

			set
			{
				m_alphaX = value;	
			}

		}

		private float m_alphaY;

		private float alphaY
		{
			get
			{
				return m_alphaY;
			}

			set
			{
				m_alphaY = value;	
			}

		}
		private float m_x0;

		private float x0
		{
			get
			{
				return m_x0;
			}

			set
			{
				m_x0 = value;	
			}

		}

		private float m_y0;

		private float y0
		{
			get
			{
				return m_y0;
			}

			set
			{
				m_y0 = value;	
			}

		}
		#endregion

		#region Internal Transformation Functions
		private float AffineX(double x)
		{
			return (float)(alphaX*x + x0);
		}
		private float AffineY(double y)
		{
			return (float)(alphaY*y + y0);
		}

		public float RevAffineX(double x)
		{
			return (float)(x - x0)/alphaX;
		}
		public float RevAffineY(double y)
		{
			return (float)(y - y0)/alphaY;
		}

		private float ShrinkX(double x)
		{
			return (float)(alphaX*x);
		}
		private float ShrinkY(double y)
		{
			return (float)(Math.Abs(alphaY)*y);
		}
		#endregion

		#region Internal Drawing Functions

        private void SetQuality(System.Drawing.Graphics g)
        {
            g.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceOver;
            g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.High;
            g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
            g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.ClearTypeGridFit;            
        }

        public abstract class Marker
        {
            public abstract string MarkerType { get; }

            public abstract void Draw(System.Drawing.Graphics g, System.Drawing.Pen pen, System.Drawing.Brush brush, uint size, float x, float y);

            public class None : Marker
            {
                public override string MarkerType { get { return "None"; } }
                public override void Draw(Graphics g, System.Drawing.Pen pen, System.Drawing.Brush brush, uint size, float x, float y) { }
            }

            public class Circle : Marker
            {
                public override string MarkerType { get { return "Circle"; } }
                public override void Draw(Graphics g, System.Drawing.Pen pen, System.Drawing.Brush brush, uint size, float x, float y) 
                { g.DrawEllipse(pen, x - size * 0.5f, y - size * 0.5f, size, size); }
            }

            public class Square : Marker
            {
                public override string MarkerType { get { return "Square"; } }
                public override void Draw(Graphics g, System.Drawing.Pen pen, System.Drawing.Brush brush, uint size, float x, float y) 
                { g.DrawRectangle(pen, x - size * 0.5f, y - size * 0.5f, size, size); }
            }

            public class UpTriangle : Marker
            {
                public override string MarkerType { get { return "UpTriangle"; } }
                public override void Draw(Graphics g, System.Drawing.Pen pen, System.Drawing.Brush brush, uint size, float x, float y)
                { g.DrawPolygon(pen, new PointF[] { new PointF(x, y - 0.5f * size), new PointF(x + 0.5f * size, y + 0.2f * size), new PointF(x - 0.5f * size, y + 0.2f * size), new PointF(x, y - 0.5f * size) }); }
            }

            public class DownTriangle : Marker
            {
                public override string MarkerType { get { return "DownTriangle"; } }
                public override void Draw(Graphics g, System.Drawing.Pen pen, System.Drawing.Brush brush, uint size, float x, float y)
                { g.DrawPolygon(pen, new PointF[] { new PointF(x, y + 0.5f * size), new PointF(x + 0.5f * size, y - 0.2f * size), new PointF(x - 0.5f * size, y - 0.2f * size), new PointF(x, y + 0.5f * size) }); }
            }

            public class Diamond : Marker
            {
                public override string MarkerType { get { return "Diamond"; } }
                public override void Draw(Graphics g, System.Drawing.Pen pen, System.Drawing.Brush brush, uint size, float x, float y)
                { g.DrawPolygon(pen, new PointF[] { new PointF(x, y + 0.5f * size), new PointF(x + 0.5f * size, y), new PointF(x, y - 0.5f * size), new PointF(x - 0.5f * size, y), new PointF(x, y + 0.5f * size) }); }
            }

            public class FilledCircle : Marker
            {
                public override string MarkerType { get { return "FilledCircle"; } }
                public override void Draw(Graphics g, System.Drawing.Pen pen, System.Drawing.Brush brush, uint size, float x, float y)
                { g.FillEllipse(brush, x - size * 0.5f, y - size * 0.5f, size, size); }
            }

            public class FilledSquare : Marker
            {
                public override string MarkerType { get { return "FilledSquare"; } }
                public override void Draw(Graphics g, System.Drawing.Pen pen, System.Drawing.Brush brush, uint size, float x, float y)
                { g.FillRectangle(brush, x - size * 0.5f, y - size * 0.5f, size, size); }
            }

            public class FilledUpTriangle : Marker
            {
                public override string MarkerType { get { return "FilledUpTriangle"; } }
                public override void Draw(Graphics g, System.Drawing.Pen pen, System.Drawing.Brush brush, uint size, float x, float y)
                { g.FillPolygon(brush, new PointF[] { new PointF(x, y - 0.5f * size), new PointF(x + 0.5f * size, y + 0.2f * size), new PointF(x - 0.5f * size, y + 0.2f * size) }); }
            }

            public class FilledDownTriangle : Marker
            {
                public override string MarkerType { get { return "FilledDownTriangle"; } }
                public override void Draw(Graphics g, System.Drawing.Pen pen, System.Drawing.Brush brush, uint size, float x, float y)
                { g.FillPolygon(brush, new PointF[] { new PointF(x, y + 0.5f * size), new PointF(x + 0.5f * size, y - 0.2f * size), new PointF(x - 0.5f * size, y - 0.2f * size) }); }
            }

            public class FilledDiamond : Marker
            {
                public override string MarkerType { get { return "FilledDiamond"; } }
                public override void Draw(Graphics g, System.Drawing.Pen pen, System.Drawing.Brush brush, uint size, float x, float y)
                { g.FillPolygon(brush, new PointF[] { new PointF(x, y + 0.5f * size), new PointF(x + 0.5f * size, y), new PointF(x, y - 0.5f * size), new PointF(x - 0.5f * size, y) }); }
            }

            internal static Marker[] s_KnownMarkers = new Marker[]
            {
                new Marker.None(), 
                new Marker.Circle(), new Marker.Square(), new Marker.UpTriangle(), new Marker.DownTriangle(), new Marker.Diamond(),
                new Marker.FilledCircle(), new Marker.FilledSquare(), new Marker.FilledUpTriangle(), new Marker.FilledDownTriangle(), new Marker.FilledDiamond()
            };

            internal static Marker GetMarker(string marker)
            {
                foreach (var m in s_KnownMarkers)
                    if (String.Compare(marker, m.MarkerType, true) == 0)
                        return m;
                throw new Exception("Unsupported marker");
            }

            public static string[] KnownMarkers
            {
                get
                {
                    string[] s = new string[s_KnownMarkers.Length];
                    int i;
                    for (i = 0; i < s_KnownMarkers.Length; i++)
                        s[i] = s_KnownMarkers[i].MarkerType;
                    return s;
                }
            }
        }

		private System.Drawing.Color [] HueIndexTable = InitRGBContinuousHueIndexTable();

		private static System.Drawing.Color [] InitRGBContinuousHueIndexTable()
		{
			System.Drawing.Color [] c = new System.Drawing.Color[256];
			int i;
			double x, y;
/*
			c[0] = System.Drawing.Color.FromArgb(255, 255, 255);
			for (i = 1; i < 51; i++)
				c[i] = System.Drawing.Color.FromArgb((51 - i) * 5, 0, 255);
			for (i = 51; i < 102; i++)
				c[i] = System.Drawing.Color.FromArgb(0, (i - 51) * 5, 255);				
			for (i = 102; i < 153; i++)
				c[i] = System.Drawing.Color.FromArgb(0, 255, (152 - i) * 5);				
			for (i = 153; i < 204; i++)
				c[i] = System.Drawing.Color.FromArgb((i - 152) * 5, 255, 0);				
			for (i = 204; i < 256; i++)
				c[i] = System.Drawing.Color.FromArgb(255, (255 - i) * 5, 0);
*/				
/*
			for (i = 0; i < 64; i++)
				c[i] = System.Drawing.Color.FromArgb(0, i * 4, 255);
			for (i = 64; i < 128; i++)
				c[i] = System.Drawing.Color.FromArgb(0, 255, (127 - i) * 4);
			for (i = 128; i < 192; i++)
				c[i] = System.Drawing.Color.FromArgb((i - 128) * 4, 255, 0);
			for (i = 192; i < 256; i++)
				c[i] = System.Drawing.Color.FromArgb(255, (255 - i) * 4, 0);
*/
			for (i = 0; i < 64; i++)
			{
				x = i / 64.0;
				y = x * x * (3 - 2 * x);
				c[i] = System.Drawing.Color.FromArgb(0, (int)(y * 255), 255);
			}
			for (i = 0; i < 64; i++)
			{
				x = i / 64.0;
				y = x * x * (3 - 2 * x);
				c[i + 64] = System.Drawing.Color.FromArgb(0, 255, (int)(255 * (1.0 - y)));
			}
			for (i = 0; i < 64; i++)
			{
				x = i / 64.0;
				y = x * x * (3 - 2 * x);
				c[i + 128] = System.Drawing.Color.FromArgb((int)(y * 255), 255, 0);
			}
			for (i = 0; i < 64; i++)
			{
				x = i / 64.0;
				y = x * x * (3 - 2 * x);
				c[i + 192] = System.Drawing.Color.FromArgb(255, (int)(255 * (1.0 - y)), 0);
			}
			return c;
		}

		private static System.Drawing.Color [] InitGreyContinuousIndexTable()
		{
			System.Drawing.Color [] c = new System.Drawing.Color[256];
			int i;
			for (i = 0; i < 256; i++)
				c[i] = System.Drawing.Color.FromArgb(255 - i, 255 - i, 255 - i);
			return c;
		}

		private static System.Drawing.Color [] InitGreyFlatIndexTable()
		{
			System.Drawing.Color [] c = new System.Drawing.Color[256];
			int i;
			c[0] = System.Drawing.Color.FromArgb(255, 255, 255);
			for (i = 1; i < 256; i++)
				c[i] = System.Drawing.Color.FromArgb((255 - i) / 16 * 16, (255 - i) / 16 * 16, (255 - i) / 16 * 16);
			return c;
		}

		private static System.Drawing.Color [] InitFlatHueIndexTable()
		{
			System.Drawing.Color [] c = new System.Drawing.Color[256];
			int i;
			c[0] = System.Drawing.Color.FromArgb(255, 255, 255);
			for (i = 1; i < 16; i++)
				c[i] = System.Drawing.Color.FromArgb(128, 128, 192);
			for (i = 16; i < 32; i++)
				c[i] = System.Drawing.Color.FromArgb(128, 192, 128);
			for (i = 32; i < 48; i++)
				c[i] = System.Drawing.Color.FromArgb(192, 128, 128);
			for (i = 48; i < 64; i++)
				c[i] = System.Drawing.Color.FromArgb(128, 192, 192);
			for (i = 64; i < 80; i++)
				c[i] = System.Drawing.Color.FromArgb(192, 128, 192);
			for (i = 80; i < 96; i++)
				c[i] = System.Drawing.Color.FromArgb(192, 192, 128);
			for (i = 96; i < 112; i++)
				c[i] = System.Drawing.Color.FromArgb(192, 192, 192);
			for (i = 112; i < 128; i++)
				c[i] = System.Drawing.Color.FromArgb(128, 96, 64);				
			for (i = 128; i < 144; i++)
				c[i] = System.Drawing.Color.FromArgb(64, 96, 128);				
			for (i = 144; i < 160; i++)
				c[i] = System.Drawing.Color.FromArgb(0, 0, 255);				
			for (i = 160; i < 176; i++)
				c[i] = System.Drawing.Color.FromArgb(0, 255, 255);				
			for (i = 176; i < 192; i++)
				c[i] = System.Drawing.Color.FromArgb(255, 0, 255);								
			for (i = 192; i < 208; i++)
				c[i] = System.Drawing.Color.FromArgb(0, 255, 0);
			for (i = 208; i < 224; i++)
				c[i] = System.Drawing.Color.FromArgb(255, 255, 0);				
			for (i = 224; i < 256; i++)
				c[i] = System.Drawing.Color.FromArgb(255, 0, 0);				
			return c;
		}

		private void BilinearHueGradient(System.Drawing.Graphics b, int x, int y, int w, int h, double lt_hi, double rt_hi, double lb_hi, double rb_hi, bool lt_valid, bool rt_valid, bool lb_valid, bool rb_valid)
		{
			int yi, xi;
			double ih = 1.0 / h;
			double iw = 1.0 / w;
			double lbt = lb_hi - lt_hi;
			double rbt = rb_hi - rt_hi;
			double lt_v = lt_valid ? 1.0 : 0.0;
			double lb_v = lb_valid ? 1.0 : 0.0;
			double rt_v = rt_valid ? 1.0 : 0.0;
			double rb_v = rb_valid ? 1.0 : 0.0;
			double vlbt = lb_v - lt_v;
			double vrbt = rb_v - rt_v;
			for (yi = 0; yi < h; yi++)
			{
				double li = lbt * yi * ih + lt_hi;
				double ri = rbt * yi * ih + rt_hi;
				double lr = ri - li;
				double lv = vlbt * yi * ih + lt_v;
				double rv = vrbt * yi * ih + rt_v;
				double lrv = rv - lv;
				for (xi = 0; xi < w; xi++)				
				{
					double s = (lrv * xi * iw + lv);
					System.Drawing.Color c = HueIndexTable[Math.Min(255, Math.Max(0, (int)Math.Round((lr * xi * iw + li) * 255)))];
					System.Drawing.Color cs = System.Drawing.Color.FromArgb((int)(c.R * s + 255 * (1 - s)), (int)(c.G * s + 255 * (1 - s)), (int)(c.B * s + 255 * (1 - s)));
					b.FillRectangle(new System.Drawing.SolidBrush(cs), xi + x, yi + y, 1, 1);
				}
			}
		}

		private void BilinearHueGradient(System.Drawing.Graphics b, int x, int y, int w, int h, double lt_hi, double rt_hi, double lb_hi, double rb_hi)
		{
			int yi, xi;
			double ih = 1.0 / h;
			double iw = 1.0 / w;
			double lbt = lb_hi - lt_hi;
			double rbt = rb_hi - rt_hi;
			for (yi = 0; yi < h; yi++)
			{
				double li = lbt * yi * ih + lt_hi;
				double ri = rbt * yi * ih + rt_hi;
				double lr = ri - li;
				for (xi = 0; xi < w; xi++)				
					b.FillRectangle(new System.Drawing.SolidBrush(HueIndexTable[Math.Min(255, Math.Max(0, (int)Math.Round((lr * xi * iw + li) * 255)))]), xi + x, yi + y, 1, 1);
			}
		}

        int m_FunctionSteps = 3;

		private void SetBorders(Graphics gPB, double MaxX, double MinX, double MaxY, double MinY,
			int Height, int Width, ref float AggX, ref float AggY, ref float LengthX, ref float LengthY,
			ref float AggFontX, ref float AggFontY, ref string StrY)
		{

			int i=0;
			double k=1;

			//Aggiunge un po' di spazio ai bordi
			//per non far disegnare i dati fino
			//alle estremita': il 10% del range
			AggX =(float)(0.1 * (MaxX - MinX));
			AggY =(float)(0.1 * (MaxY - MinY));
			LengthX = (float)(MaxX - MinX);
			LengthY = (float)(MaxY - MinY);

			AggFontY = gPB.MeasureString("0.0",m_LabelFont).Height;
			alphaY = (-Height+AggFontY)/ (LengthY+2*AggY);
			AggFontY = AggFontY /Math.Abs(alphaY);
			y0 = (float)(Math.Abs(alphaY)*(MaxY+AggY-AggFontY));

			do
			{
				i++;
				k=k*0.1;
			} while (AggFontY<k);
			StrY="F"+i;

			AggFontX = gPB.MeasureString(MaxY.ToString(StrY),m_LabelFont).Width;
			if (gPB.MeasureString(MinY.ToString(StrY),m_LabelFont).Width> AggFontX) AggFontX=gPB.MeasureString(MinY.ToString(StrY),m_LabelFont).Width;
			alphaX = (Width-AggFontX) / (LengthX+2*AggX);
			AggFontX = AggFontX /alphaX;
			x0  = (float)(alphaX*(-MinX+0.5*AggX+AggFontX));

            m_FunctionSteps = (int)(Width / 2 + 1);

		}



		private void DrawXAxis(int BinNumber, double AxisYPos, double MaxX, double MinX, double MaxY, double MinY, double AggX, double AggY, double AggFontX, double AggFontY, Graphics gPB)
		{
			double tickstep, mintick;
			double range = MaxX - MinX;
			int divfactor = 0;
			int steps, s;			
			if (MaxX <= MinX) MaxX = MinX + 1.0;			
			tickstep = Math.Pow(10.0, Math.Ceiling(Math.Log10(MaxX - MinX)));
            if (tickstep <= 0.0) tickstep = 1.0;
			while (tickstep >= range)
			{
				divfactor = (++divfactor % 3);
				switch (divfactor)
				{
					case 1:		tickstep *= 0.5; break;
					case 2:		tickstep *= 0.4; break;
					case 0:		tickstep *= 0.5; break;
				}
			}
			TickMark [] Ticks = null, OldTicks = null;
			double X = 0, Y = 0, TickY = 0, v;
			int prec;
			string label;

			Y = AffineY(AxisYPos);
			gPB.DrawLine(myPen, AffineX(MinX), (float)Y, AffineX(MaxX), (float)Y);
			TickY = AffineY(AxisYPos - AggY) - AffineY(AxisYPos);
			do
			{
				OldTicks = Ticks;
				mintick = Math.Ceiling(MinX / tickstep) * tickstep;
				prec = (int)Math.Ceiling(-Math.Log10(tickstep)); if (prec < 0) prec = 0;
				steps = (int)Math.Ceiling((MaxX - mintick) / tickstep);
				Ticks = new TickMark[steps];
				for (s = 0; s < steps; s++)
				{				
					v = mintick + s * tickstep;
					X = AffineX(v);				
					label = v.ToString("F" + prec);
					System.Drawing.SizeF size = gPB.MeasureString(label, m_LabelFont);
					Ticks[s] = new TickMark((int)Math.Round(X), (int)Y, (int)size.Width + 2, (int)size.Height + 2, label);
					if (s > 0 && OldTicks != null)
					{
						if (Ticks[s].Rect.IntersectsWith(Ticks[s - 1].Rect))
						{
							Ticks = OldTicks;
							break;
						}
					}
				}
				divfactor = (++divfactor % 3);
				switch (divfactor)
				{
					case 1:		tickstep *= 0.5; break;
					case 2:		tickstep *= 0.4; break;
					case 0:		tickstep *= 0.5; break;
				}
			}
			while (Ticks != OldTicks);
			steps = Ticks.Length;
			for (s = 0; s < (steps - 1); s++)
			{								
				int tx, ty;
				tx = (Ticks[s].Rect.Left + Ticks[s].Rect.Right + Ticks[s + 1].Rect.Left + Ticks[s + 1].Rect.Right) / 4;
				ty = (Ticks[s].Rect.Top + Ticks[s].Rect.Bottom) / 2 + 1;
				gPB.DrawLine(myPen, tx, ty, tx, ty + (int)TickY / 3);
			}
			for (s = 0; s < steps; s++)							
			{								
				int tx, ty;
				gPB.DrawLine(myPen, (Ticks[s].Rect.Left + Ticks[s].Rect.Right) / 2, (Ticks[s].Rect.Top + Ticks[s].Rect.Bottom) / 2 + 1,
					tx = (Ticks[s].Rect.Left + Ticks[s].Rect.Right) / 2, ty = (Ticks[s].Rect.Top + Ticks[s].Rect.Bottom) / 2 + (int)TickY / 2);
				Ticks[s].Rect.Offset(tx - Ticks[s].Rect.Left - Ticks[s].Rect.Width / 2, ty - Ticks[s].Rect.Top);
				gPB.DrawString(Ticks[s].Text, m_LabelFont, myBrush, Ticks[s].Rect);
			}
		}

		private void DrawYAxis( double AxisXPos,
			/*double MaxX, double MinX,*/ 
			double MaxY, double MinY, 
			double AggX, double AggY, double AggFontX, double AggFontY, 
			string StrY, Graphics gPB)
		{

			double CuX=0, CuY, OldY;
			int i=0;
			float tmpX1, tmpY1, tmpX2; 
			string tmpString;
			

			//double  SpazioLabelY = Math.Pow(10.0, Math.Ceiling(Math.Log10(AggFontY)));
			//double SpazioLabelY = Fitting.ExtendedRound(AggFontY,1,NumericalTools.RoundOption.CeilingRound);
			double SpazioLabelY=1;
			if (AggFontY<SpazioLabelY)
			{
				SpazioLabelY*=0.1;
				while(AggFontY<SpazioLabelY)
				{
					SpazioLabelY*=0.1;
				}
				
				if(SpazioLabelY*2>AggFontY)
				{
					SpazioLabelY*=2;
				}
				else if(SpazioLabelY*5>AggFontY)
				{
					SpazioLabelY*=5;
			
				}
				else if(SpazioLabelY*10>AggFontY)
				{
					SpazioLabelY*=10;
				
				};
			}
			else
			{
				SpazioLabelY*=10;
				while(AggFontY>SpazioLabelY)
				{
					SpazioLabelY*=10;
				}
				
				if(SpazioLabelY/10>AggFontY)
				{
					SpazioLabelY/=10;
				}
				else if(SpazioLabelY/5>AggFontY)
				{
					SpazioLabelY/=5;
			
				}
				else if(SpazioLabelY/2>AggFontY)
				{
					SpazioLabelY/=2;
				
				};
			
			}

			if (SpazioLabelY <= 0.0) StrY = "F0";
			else StrY = "F" + ((int)Math.Max(0, -Math.Floor(Math.Log10(SpazioLabelY)))).ToString();
			
			float LengthY = (float)(MaxY-MinY);
			float CentroY = (float)((MaxY+MinY)/2);
			double MinAsseY = CentroY - 0.52*LengthY;
			//double MaxAsseY = CentroY + 0.52*LengthY;
			double MaxAsseY = CentroY + 0.5*LengthY;

			gPB.DrawLine(myPen, AffineX(AxisXPos), AffineY(MinAsseY), 
				AffineX(AxisXPos), AffineY(MaxAsseY));

			double MinPartY = Math.Floor(MinY / SpazioLabelY) * SpazioLabelY;
			OldY = MinPartY;//MinY;
			CuY = MinPartY;//MinY;
			System.Drawing.PointF myPt = new System.Drawing.PointF();

			SpazioLabelY *= 0.5;
			do
			{
				i++;
				tmpX1 = (float)(-AggX / ((i % 2 != 1) ? 3 : 4) + AxisXPos);
				tmpY1 = (float)(MinPartY/*MinY*/ + i * SpazioLabelY);
				tmpX2 = (float)(AxisXPos);
				/*
				 *CuY va comunque aggiornato anche se non viene scritto
				 *perché da esso dipende l'uscita dal loop ed il fatto che
				 *delle tacche non vengano scritte oltre l'asse Y
				 */
				CuY = MinPartY/*MinY*/ + i * SpazioLabelY ;
				//QUESTO E' IL PUNTO DOVE ESCE				
				if ((CuY - MinY) >= LengthY *0.95) break;
				gPB.DrawLine(myPen, AffineX(tmpX1), AffineY(tmpY1), 
					AffineX(tmpX2), AffineY(tmpY1));
				
				
				if(i%2 != 1)
				{
					CuX = AxisXPos - AggFontX - AggX / 3;
					if (CuY - AggFontY/2 >= OldY)
					{
						tmpString = CuY.ToString(StrY);

						myPt.X = AffineX(CuX);
						myPt.Y = AffineY(CuY+ AggFontY/2);
						gPB.DrawString(tmpString, m_LabelFont, myBrush, myPt);
						OldY = CuY - AggFontY/2;
					};
				};
			} while ((CuY - MinY) < LengthY * 0.95);

		}


		// degree = m_ScatterFit oppure degree = m_HistoFit
		private void Fit_Plot_Scatter(double[] X_Mean, double[] Y_Mean,
			int[] Ent, double[] SY_Vec, 
			double MaxX, double MinX, 
			double MaxY, double MinY,
			short degree, Graphics gPB)
		{

			double a=0, b=0, dum=0;
			int j=0, i,n;
			n = X_Mean.GetLength(0);


			for(i=0; i<n;i++)
				//if(Ent[i]>0 && ( (X_Mean[i]-m_DX/2)>=MinX) && ( (X_Mean[i]+m_DX/2)<=MaxX)) j++;
				if(Ent[i]>0 ) j++;

			double[] tx = new double[j];
			double[] ty = new double[j];
			double[] sy = new double[j];

			j=0;
			for(i=0; i<n;i++)
				//if(Ent[i]>0 && ( (X_Mean[i]-m_DX/2)>=MinX) && ( (X_Mean[i]+m_DX/2)<=MaxX)) 
				if(Ent[i]>0 ) 
				{
					tx[j] = X_Mean[i];
					ty[j] = Y_Mean[i];
					sy[j] = SY_Vec[i];
					j++;
				};


			m_FitPar= new double [degree+1];
			m_ParDescr= new string [degree+1];

			if (degree==1) 
			{
				if (m_LinearFitWE) 
					Fitting.LinearFitDE(tx, ty, sy, ref a, ref b, ref dum);
				else
					Fitting.LinearFitSE(tx, ty, ref a, ref b, ref dum, ref dum, ref dum, ref dum, ref dum);
				gPB.DrawLine(myRedPen, AffineX(MinX), AffineY(a*MinX +b), AffineX(MaxX), AffineY(a*MaxX+b)); 
				m_FitPar[0]=b; m_ParDescr[0]="Constant";
				m_FitPar[1]=a; m_ParDescr[1]="Linear";
			}
			else if (degree>1)
			{

				double ccorr=0;
				double[] of = new double[degree];
				Fitting.PolynomialFit(tx,ty,degree, ref of, ref ccorr);

				for(i=0; i< degree+1;i++)
				{
					m_FitPar[i]=of[i]; 
					m_ParDescr[i]="a["+i+"]";
				};
				double[] xpl = new double[2];
				double[] ypl = new double[2];
				n = 100;
				for(i=0; i<n-1;i++)
				{
					xpl[0]=MinX+ (MaxX-MinX)*i/99;							
					xpl[1]=MinX+ (MaxX-MinX)*(i+1)/99;							
					ypl[0]=0;
					ypl[1]=0;
					for(j=0; j< degree+1;j++)
					{
						ypl[0] += Math.Pow(xpl[0],j)*of[j];
						ypl[1] += Math.Pow(xpl[1],j)*of[j];
					};
					if (ypl[0]>MinY && ypl[0]<MaxY && ypl[1]> MinY && ypl[1]<MaxY)
						gPB.DrawLine(myRedPen, AffineX(xpl[0]),AffineY(ypl[0]),AffineX(xpl[1]),AffineY(ypl[1]));
				};
			};
		
		}


		private void Gauss_Fit_Histo(double MaxX, double MinX,
			double[] X_Mean, double[] Y_Vec, 
			double[] N_Y_Vec, Graphics gPB)
		{

			double fChi2 , CFactor=0;
			//double tmpX1=0, tmpY1=0;
			short fres;
			int i,j,cc=0;
			double[] Pars = new double[3];
			int[] Iter= new int[2];
			int n =X_Mean.GetLength(0);
			double[] tmpvecx;

			//il fit verrà fatto sempre su tutti i bins, disegnati o no...
			//forse posso anche metterci l'opzione solo sui bins disegnati...
			if(m_FittingOnlyDataInPlot)
			{
				for(j = 0; j<m_VecX.GetLength(0); j++) if (m_VecX[j]>MinX && m_VecX[j]<MaxX ) cc++;

				tmpvecx= new double[cc];
				cc=0;
				for(j = 0; j<m_VecX.GetLength(0); j++)
					if (m_VecX[j]>MinX && m_VecX[j]<MaxX )
					{
						tmpvecx[cc]=m_VecX[j];
						cc++;
					};

				Fitting.LM_GaussianRegression(tmpvecx, m_DX, 0.0000000001, 10, 5000, 10, out Pars, out Iter, out fChi2, out fres);
			}
			else
			{
				Fitting.LM_GaussianRegression(m_VecX, m_DX, 0.0000000001, 10, 5000, 10, out Pars, out Iter, out fChi2, out fres);
			
			};

			if (fres != 1)
			{
				double[] tmppar = new double[2];
				string[] tmppardescr = new string[2];
				for(i=0; i<2;i++)
				{
					tmppar[i]=m_FitPar[i];
					tmppardescr[i]=m_ParDescr[i];
				};
				m_FitPar= new double [2];
				m_ParDescr= new string [2];
				m_ParDescr=(string[])tmppardescr.Clone();
				m_FitPar=(double[])tmppar.Clone();
			}
			else
			{
				for( i = 0; i< n;i++)
					if(Y_Vec[i] != 0)
					{
						CFactor = Y_Vec[i] / N_Y_Vec[i];
						break;
					};

				for( i = 0 ; i<n; i++)
					if (((X_Mean[i] - m_DX / 2)>= MinX) && ((X_Mean[i] + m_DX / 2)<= MaxX))
					{
						PointF [] tmpP = new PointF[4];
						tmpP[0].X = (float)(X_Mean[i] - m_DX / 2);
						tmpP[1].X = (float)(X_Mean[i] - m_DX / 4);
						tmpP[2].X = (float)(X_Mean[i] + m_DX / 4);
						tmpP[3].X = (float)(X_Mean[i] + m_DX / 2);
						for(j=0; j<4; j++) 
						{							
							tmpP[j].Y = AffineY(CFactor * Pars[2] * Math.Exp(-(tmpP[j].X - Pars[0])*(tmpP[j].X - Pars[0]) / (2 * Pars[1] * Pars[1])));
							tmpP[j].X = AffineX(tmpP[j].X);
						}
						gPB.DrawLines(myPen, tmpP);

					};
				m_ParDescr[2]="Mean Fit";
				m_ParDescr[3]="Sigma";
				m_FitPar[2]=Pars[0];
				m_FitPar[3]=Pars[1];
			};
		
		}


		private void InvGauss_Fit_Histo(double MaxX, double MinX,
			double[] X_Mean, double[] Y_Vec, 
			double[] N_Y_Vec, Graphics gPB)
		{

			double fChi2 , CFactor=0;
			//double tmpX1=0, tmpY1=0;
			short fres;
			int i,j,cc=0;
			double[] Pars = new double[3];
			int[] Iter= new int[2];
			int n =X_Mean.GetLength(0);
			double[] tmpvecx;

			//il fit verrà fatto sempre su tutti i bins, disegnati o no...
			//forse posso anche metterci l'opzione solo sui bins disegnati...
			if(m_FittingOnlyDataInPlot)
			{
				for(j = 0; j<m_VecX.GetLength(0); j++) if (m_VecX[j]>MinX && m_VecX[j]<MaxX ) cc++;

				tmpvecx= new double[cc];
				cc=0;
				for(j = 0; j<m_VecX.GetLength(0); j++)
					if (m_VecX[j]>MinX && m_VecX[j]<MaxX )
					{
						tmpvecx[cc]=m_VecX[j];
						cc++;
					};

				Fitting.LM_InverseGaussianRegression(tmpvecx, m_DX, 0.0000000001, 10, 5000, 10, out Pars, out Iter, out fChi2, out fres);
			}
			else
			{
				Fitting.LM_InverseGaussianRegression(m_VecX, m_DX, 0.0000000001, 10, 5000, 10, out Pars, out Iter, out fChi2, out fres);
			
			};

			if (fres != 1)
			{
				double[] tmppar = new double[2];
				string[] tmppardescr = new string[2];
				for(i=0; i<2;i++)
				{
					tmppar[i]=m_FitPar[i];
					tmppardescr[i]=m_ParDescr[i];
				};
				m_FitPar= new double [2];
				m_ParDescr= new string [2];
				m_ParDescr=(string[])tmppardescr.Clone();
				m_FitPar=(double[])tmppar.Clone();
			}
			else
			{
				for( i = 0; i< n;i++)
					if(Y_Vec[i] != 0)
					{
						CFactor = Y_Vec[i] / N_Y_Vec[i];
						break;
					};

				for( i = 0 ; i<n; i++)
					if (((X_Mean[i] - m_DX / 2)>= MinX) && ((X_Mean[i] + m_DX / 2)<= MaxX))
					{
						PointF [] tmpP = new PointF[4];
						tmpP[0].X = (float)(X_Mean[i] - m_DX / 2);
						tmpP[1].X = (float)(X_Mean[i] - m_DX / 4);
						tmpP[2].X = (float)(X_Mean[i] + m_DX / 4);
						tmpP[3].X = (float)(X_Mean[i] + m_DX / 2);
						for(j=0; j<4; j++) 
						{
							tmpP[j].Y = AffineY(CFactor * (Pars[2]/(double)(tmpP[j].X*tmpP[j].X)) * Math.Exp(-(1.0/(float)tmpP[j].X - 1.0/(float)Pars[0])*(1.0/(float)tmpP[j].X - 1.0/(float)Pars[0]) / (float)(Pars[1] * Pars[1])));
							tmpP[j].X = AffineX(tmpP[j].X);
						}
						gPB.DrawLines(myPen, tmpP);

					};
				for(i=2;i<5;i++)
				{
					m_FitPar[i]=Pars[i-2]; 
					m_ParDescr[i]="a["+(i-2)+"]";
				};
			};
		
		}


		private void OverlayFunction(double MaxX, double MinX, 
			double MaxY, double MinY, 
			Graphics gPB)
		{
            myRedPen.Width = m_PlotThickness;
			Function f = null;
			f= new CStyleParsedFunction(m_Function);
			int npar = f.ParameterList.Length;
			if (npar>1 || MaxX<=MinX || MaxY <= MinY) return;
			double tmpX1, tmpY1, tmpX2, tmpY2;

			if(npar==1)
			{
				//int fs = (int)(gPB.VisibleClipBounds.Width / 2);
                //int fs = (int)(gPB.ClipBounds.Width / 2);                
                int fs = m_FunctionSteps;
				for(int i = 0; i < fs; i++) 
				{
					f[0] = MinX + (MaxX - MinX) * i / (fs - 1);
					tmpX1 = f[0];
					tmpY1 = f.Evaluate();
					f[0] = MinX + (MaxX - MinX) * (i + 1) / (fs - 1);
					tmpX2 = f[0];
					tmpY2 = f.Evaluate();
					gPB.DrawLine(myRedPen, AffineX(tmpX1), AffineY(tmpY1), AffineX(tmpX2), AffineY(tmpY2));
				};
			}
			else
			{
				tmpX1=MinX;
				tmpY1=f.Evaluate();
				tmpX2=MaxX;
				tmpY2=f.Evaluate();
				gPB.DrawLine(myRedPen, AffineX(tmpX1), AffineY(tmpY1), AffineX(tmpX2), AffineY(tmpY2));
			};

		}

		#endregion
	}
}
