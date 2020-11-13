using System;
using System.Collections;
using System.Drawing;
using System.Security;
[assembly:AllowPartiallyTrustedCallers]

namespace GDI3D
{
	namespace Plot
	{
		/// <summary>
		/// A GDI-based 3D Plot.
		/// </summary>
		public class Plot
		{
			internal System.Collections.ArrayList m_Lines = new System.Collections.ArrayList();
			internal System.Collections.ArrayList m_Points = new System.Collections.ArrayList();

			internal System.Drawing.Color m_BackColor = Color.Black;
			/// <summary>
			/// Gets/sets the back color of the plot.
			/// </summary>
			public System.Drawing.Color BackColor
			{
				get { return m_BackColor; }
				set 
				{
					m_BackColor = value;
					if (m_AutoRender) Render();
				}
			}

			/// <summary>
			/// Initializes an empty plot.
			/// </summary>
			public Plot()
			{
				//
				// TODO: Add constructor logic here
				//
				BufferBitmap = new Bitmap(200, 200, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
				BufferG = System.Drawing.Graphics.FromImage(BufferBitmap);
                BufferG.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;

			}

            /// <summary>
            /// Initializes an empty plot.
            /// </summary>
            public Plot(int width, int height)
            {
                //
                // TODO: Add constructor logic here
                //
                BufferBitmap = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                BufferG = System.Drawing.Graphics.FromImage(BufferBitmap);
                BufferG.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
            }

            /// <summary>
			/// Clears the plot.
			/// </summary>
			public void Clear()
			{
				m_Lines.Clear();
				m_Points.Clear();
				m_Changed = true;
				if (m_AutoRender) Render();
			}

			/// <summary>
			/// Deletes all elements associated with the specified owner.
			/// </summary>
			/// <param name="owner">Owner whose graphical representation should be deleted.</param>
			public bool DeleteWithOwner(object owner)
			{
                bool found = false;
                int i;
                for (i = 0; i < m_Lines.Count; i++) if (System.Object.ReferenceEquals(((Line)m_Lines[i]).Owner, owner)) { m_Lines.RemoveAt(i--); found = true; }
                for (i = 0; i < m_Points.Count; i++) if (System.Object.ReferenceEquals(((Point)m_Points[i]).Owner, owner)) { m_Points.RemoveAt(i--); found = true; }
                m_Changed = found;
                return found;
            }

            /// <summary>
            /// Changes the color of all elements associated with the specified owner.
            /// </summary>
            /// <param name="owner">Owner whose graphical representation should be changed.</param>
            /// <param name="r">the new red component.</param>
            /// <param name="g">the new green component.</param>
            /// <param name="b">the new blue component.</param>
            /// <param name="preserveneutral">if <c>true</c>, the neutral (grey) component is preserved; with this parameter set, objects tend to become lighter.</param>
            public bool RecolorWithOwner(object owner, int r, int g, int b, bool preserveneutral)
            {
                bool found = false;
                int i;
                for (i = 0; i < m_Lines.Count; i++) if (System.Object.ReferenceEquals(((Line)m_Lines[i]).Owner, owner))
                    {
                        Line l = (Line)m_Lines[i];
                        int min = preserveneutral ? Math.Min(l.R, Math.Min(l.G, l.B)) : 0;
                        l.R = Math.Max(min, r);
                        l.G = Math.Max(min, g);
                        l.B = Math.Max(min, b);
                        found = true;
                    }
                for (i = 0; i < m_Points.Count; i++) if (System.Object.ReferenceEquals(((Point)m_Points[i]).Owner, owner))
                    {
                        Point p = (Point)m_Points[i];
                        int min = preserveneutral ? Math.Min(p.R, Math.Min(p.G, p.B)) : 0;
                        p.R = Math.Max(min, r);
                        p.G = Math.Max(min, g);
                        p.B = Math.Max(min, b);
                        found = true;
                    }
                m_Changed = found;
                return found;
            }

            /// <summary>
            /// Checks whether the plot contains references to an owner.
            /// </summary>
            /// <param name="owner">Owner whose graphical elements are being sought.</param>
            /// <returns><c>true</c> if the owner is present, <c>false</c> otherwise.</returns>
            public bool Contains(object owner)
            {
                int i;
                for (i = 0; i < m_Lines.Count; i++) if (System.Object.ReferenceEquals(((Line)m_Lines[i]).Owner, owner)) return true;
                for (i = 0; i < m_Points.Count; i++) if (System.Object.ReferenceEquals(((Point)m_Points[i]).Owner, owner)) return true;
                return false;
            }
			/// <summary>
			/// Adds a line.
			/// </summary>
			/// <param name="l">Line to be added.</param>
			public void Add(Line l)
			{
				m_Lines.Add(l);
				m_Changed = true;
				if (m_AutoRender) Render();
			}
			/// <summary>
			/// Adds a point.
			/// </summary>
			/// <param name="p">Point to be added.</param>
			public void Add(Point p)
			{
				m_Points.Add(p);			
				m_Changed = true;
				if (m_AutoRender) Render();
			}

			internal bool m_Changed = true;

			internal bool m_AutoRender;
			/// <summary>
			/// Tells whether changes should be immediately reflected to the plot.
			/// </summary>
			public bool AutoRender
			{
				get
				{
					return m_AutoRender;
				}
				set
				{
					m_AutoRender = value;
				}
			}

			internal double m_Zoom = 100.0;
			/// <summary>
			/// Gets/sets the zoom factor.
			/// </summary>
			public double Zoom
			{
				get
				{
					return m_Zoom;
				}
				set
				{
					m_Zoom = value;
					m_Changed = true;
					if (m_AutoRender) Render();
				}
			}

			internal bool m_Infinity = false;
			/// <summary>
			/// Gets/sets the infinity projection flag: if set, an isometric view instead of a perspective is produced.
			/// </summary>
			public bool Infinity
			{
				get
				{
					return m_Infinity;
				}
				set
				{
					m_Infinity = value;
					m_Changed = true;
					if (m_AutoRender) Render();
				}
			}

			internal double m_CameraPosX = 0.0;
			internal double m_CameraPosY = 0.0;
			internal double m_CameraPosZ = 100.0;
			/// <summary>
			/// Gets the camera position.
			/// </summary>
			/// <param name="X">X component</param>
			/// <param name="Y">Y component</param>
			/// <param name="Z">Z component</param>
			public void GetCameraPosition(ref double X, ref double Y, ref double Z)
			{
				X = m_CameraPosX;
				Y = m_CameraPosY;
				Z = m_CameraPosZ;
			}
			/// <summary>
			/// Sets the camera position
			/// </summary>
			/// <param name="X">X component</param>
			/// <param name="Y">Y component</param>
			/// <param name="Z">Z component</param>
			public void SetCameraPosition(double X, double Y, double Z)
			{
				m_CameraPosX = X;
				m_CameraPosY = Y;
				m_CameraPosZ = Z;
				m_CameraSpottingX = m_CameraPosX + m_CameraDirectionX * m_Distance;
				m_CameraSpottingY = m_CameraPosY + m_CameraDirectionY * m_Distance;
				m_CameraSpottingZ = m_CameraPosZ + m_CameraDirectionZ * m_Distance;
				m_Changed = true;
				if (m_AutoRender) Render();
			}

			internal double m_CameraDirectionX = 0.0;
			internal double m_CameraDirectionY = 0.0;
			internal double m_CameraDirectionZ = -1.0;
			internal double m_CameraNormalX = 0.0;
			internal double m_CameraNormalY = -1.0;
			internal double m_CameraNormalZ = 0.0;
			internal double m_CameraBinormalX = -1.0;
			internal double m_CameraBinormalY = 0.0;
			internal double m_CameraBinormalZ = 0.0;
			/// <summary>
			/// Sets the camera orientation.
			/// </summary>
			/// <param name="DX">X component of camera direction</param>
			/// <param name="DY">Y component of camera direction</param>
			/// <param name="DZ">Z component of camera direction</param>
			/// <param name="NX">X component of camera normal</param>
			/// <param name="NY">Y component of camera normal</param>
			/// <param name="NZ">Z component of camera normal</param>
			public void SetCameraOrientation(double DX, double DY, double DZ, double NX, double NY, double NZ)
			{
				double norm = Math.Sqrt(DX * DX + DY * DY + DZ * DZ);
				if (norm <= 0) return;
				norm = 1.0 / norm;
				DX *= norm;
				DY *= norm;
				DZ *= norm;
				double n = NX * DX + NY * DY + NZ * DZ;
				NX -= DX * n;
				NY -= DY * n;
				NZ -= DZ * n;
				norm = Math.Sqrt(NX * NX + NY * NY + NZ * NZ);
				if (norm <= 0) return;
				NX *= norm;
				NY *= norm;
				NZ *= norm;
				m_CameraDirectionX = DX;
				m_CameraDirectionY = DY;
				m_CameraDirectionZ = DZ;
				m_CameraNormalX = NX;
				m_CameraNormalY = NY;
				m_CameraNormalZ = NZ;
				m_CameraBinormalX = DY * NZ - DZ * NY;
				m_CameraBinormalY = DZ * NX - DX * NZ;
				m_CameraBinormalZ = DX * NY - DY * NX;
				m_Changed = true;
				if (m_AutoRender) Render();
			}
			/// <summary>
			/// Gets the camera orientation.
			/// </summary>
			/// <param name="DX">X component of camera direction</param>
			/// <param name="DY">Y component of camera direction</param>
			/// <param name="DZ">Z component of camera direction</param>
			/// <param name="NX">X component of camera normal</param>
			/// <param name="NY">Y component of camera normal</param>
			/// <param name="NZ">Z component of camera normal</param>
			/// <param name="BX">X component of camera binormal</param>
			/// <param name="BY">Y component of camera binormal</param>
			/// <param name="BZ">Z component of camera binormal</param>
			public void GetCameraOrientation(ref double DX, ref double DY, ref double DZ, ref double NX, ref double NY, ref double NZ, ref double BX, ref double BY, ref double BZ)
			{
				DX = m_CameraDirectionX;
				DY = m_CameraDirectionY;
				DZ = m_CameraDirectionZ;
				NX = m_CameraNormalX;
				NY = m_CameraNormalY;
				NZ = m_CameraNormalZ;
				BX = m_CameraBinormalX;
				BY = m_CameraBinormalY;
				BZ = m_CameraBinormalZ;
			}

			internal double m_Distance = 10.0;
			/// <summary>
			/// Gets/sets the distance from the camera.
			/// </summary>
			public double Distance
			{
				get { return m_Distance; }
				set
				{				
					m_Distance = value;
					m_CameraPosX = m_CameraSpottingX - m_CameraDirectionX * m_Distance;
					m_CameraPosY = m_CameraSpottingY - m_CameraDirectionY * m_Distance;
					m_CameraPosZ = m_CameraSpottingZ - m_CameraDirectionZ * m_Distance;
					m_Changed = true;
					if (m_AutoRender) Render();
				}
			}

            /// <summary>
            /// Gets the scene.
            /// </summary>
            public GDI3D.Scene GetScene()
            {
                System.Collections.ArrayList signatures = new System.Collections.ArrayList();
                int index, i;
                GDI3D.Scene scene = new GDI3D.Scene();
                scene.Points = new GDI3D.Point[m_Points.Count];
                for (i = 0; i < m_Points.Count; i++)
                {
                    Point p = (Point)m_Points[i];
                    GDI3D.Point gp = new GDI3D.Point(p.X, p.Y, p.Z, -1, p.R, p.G, p.B);
                    if (p.Owner != null)
                    {
                        for (index = 0; index < signatures.Count && signatures[index] != p.Owner; index++) ;
                        if (index == signatures.Count)
                            signatures.Add(p.Owner);
                        gp.Owner = index;
                    }
                    scene.Points[i] = gp;
                }
                scene.Lines = new GDI3D.Line[m_Lines.Count];
                for (i = 0; i < m_Lines.Count; i++)
                {
                    Line l = (Line)m_Lines[i];
                    GDI3D.Line gl = new GDI3D.Line(l.XF, l.YF, l.ZF, l.XS, l.YS, l.ZS, -1, l.R, l.G, l.B);
                    if (l.Owner != null)
                    {
                        for (index = 0; index < signatures.Count && signatures[index] != l.Owner; index++) ;
                        if (index == signatures.Count)
                            signatures.Add(l.Owner);
                        gl.Owner = index;
                    }
                    scene.Lines[i] = gl;
                }
                scene.OwnerSignatures = new string[signatures.Count];
                for (i = 0; i < signatures.Count; i++)
                    scene.OwnerSignatures[i] = signatures[i].ToString();
                scene.Zoom = this.m_Zoom;
                scene.BackColor = Color.FromArgb(this.BackColor.R, this.BackColor.G, this.BackColor.B);
                scene.CameraDirectionX = this.m_CameraDirectionX;
                scene.CameraDirectionY = this.m_CameraDirectionY;
                scene.CameraDirectionZ = this.m_CameraDirectionZ;
                scene.CameraNormalX = this.m_CameraNormalX;
                scene.CameraNormalY = this.m_CameraNormalY;
                scene.CameraNormalZ = this.m_CameraNormalZ;
                scene.CameraSpottingX = this.m_CameraSpottingX;
                scene.CameraSpottingY = this.m_CameraSpottingY;
                scene.CameraSpottingZ = this.m_CameraSpottingZ;
                scene.CameraDistance = this.m_Distance;
                return scene;
            }

            internal void SaveMetafile(string filename)
            {                
                System.Drawing.Graphics g = System.Drawing.Graphics.FromImage(BufferBitmap);
                IntPtr hdc = g.GetHdc();
                System.Drawing.Imaging.Metafile mf = new System.Drawing.Imaging.Metafile(filename, hdc, new Rectangle(0, 0, BufferBitmap.Width, BufferBitmap.Height), System.Drawing.Imaging.MetafileFrameUnit.Pixel);
                g.ReleaseHdc();
                g.Dispose();
                g = System.Drawing.Graphics.FromImage(mf);
                g.ScaleTransform(1.0f, 1.0f);
                Render(g);
                g.Dispose();
                mf.Dispose();
            }
			/// <summary>
			/// Saves the image in the specified format.
			/// </summary>
			/// <param name="stream">Stream to save the image to.</param>
			/// <param name="format">Image format.</param>
			public void Save(System.IO.Stream stream, System.Drawing.Imaging.ImageFormat format)
			{
				Render();
				BufferBitmap.Save(stream, format);
			}
			/// <summary>
			/// Saves the image in the specified format.
			/// </summary>
			/// <param name="filename">Name of the file to save the image to.</param>
			/// <param name="format">Image format.</param>
			public void Save(string filename, System.Drawing.Imaging.ImageFormat format)
			{
                if (filename.ToLower().EndsWith(".emf")) SaveMetafile(filename);
                else
                {
                    Render();
                    BufferBitmap.Save(filename, format);
                }
			}
			/// <summary>
			/// Saves the image in a default format.
			/// </summary>
			/// <param name="filename">Name of the file to save the image to.</param>
			public void Save(string filename)
			{
                if (filename.ToLower().EndsWith(".emf")) SaveMetafile(filename);
                else
                {
                    Render();
                    if (filename.ToLower().EndsWith(".x3l"))
                    {
                        System.IO.StreamWriter wr = null;
                        System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(GDI3D.Scene));
                        try
                        {
                            wr = new System.IO.StreamWriter(filename);
                            GDI3D.Scene scene = GetScene();
                            xmls.Serialize(wr, scene);
                            wr.Flush();
                            wr.Close();
                            wr = null;
                        }
                        catch (Exception x)
                        {
                            if (wr != null) wr.Close();
                            throw x;
                        }
                    }
                    else BufferBitmap.Save(filename);
                }
			}

			internal double m_CameraSpottingX = 0.0;
			internal double m_CameraSpottingY = 0.0;
			internal double m_CameraSpottingZ = 0.0;

			/// <summary>
			/// Gets the camera spotting point.
			/// </summary>
			/// <param name="X">X component</param>
			/// <param name="Y">Y component</param>
			/// <param name="Z">Z component</param>
			public void GetCameraSpotting(ref double X, ref double Y, ref double Z)
			{
				X = m_CameraSpottingX;
				Y = m_CameraSpottingY;
				Z = m_CameraSpottingZ;
			}
			/// <summary>
			/// Sets the camera spotting point.
			/// </summary>
			/// <param name="X">X component</param>
			/// <param name="Y">Y component</param>
			/// <param name="Z">Z component</param>
			public void SetCameraSpotting(double X, double Y, double Z)
			{
				m_CameraSpottingX = X;
				m_CameraSpottingY = Y;
				m_CameraSpottingZ = Z;
				m_CameraPosX = m_CameraSpottingX - m_CameraDirectionX * m_Distance;
				m_CameraPosY = m_CameraSpottingY - m_CameraDirectionY * m_Distance;
				m_CameraPosZ = m_CameraSpottingZ - m_CameraDirectionZ * m_Distance;
				m_Changed = true;
				if (m_AutoRender) Render();
			}
		
			internal int m_LineWidth = 1;
			/// <summary>
			/// Gets/sets the width of lines.
			/// </summary>
			public int LineWidth
			{
				get { return m_LineWidth; }
				set 
				{ 
					m_LineWidth = value; 
					if (m_AutoRender) Render();
				}
			}

			internal int m_PointSize = 5;
			/// <summary>
			/// Gets/sets the size of points;
			/// </summary>
			public int PointSize
			{
				get { return m_PointSize; }
				set 
				{ 				
					m_PointSize = value; 
					if (m_AutoRender) Render();
				}
			}

			internal int m_BorderWidth = 1;
			/// <summary>
			/// The width of border; setting this number to 0 disables borders.
			/// </summary>
			public int BorderWidth
			{
				get { return m_BorderWidth; }
				set 
				{ 
					m_BorderWidth = value; 
					if (m_AutoRender) Render();
				}
			}

			internal int m_Alpha = 128;
			/// <summary>
			/// Alpha component of all lines and points. Ranges from 0 to 1.
			/// </summary>
			public double Alpha
			{
				get { return m_Alpha / 255.0; }
				set 
				{ 
					m_Alpha = ((int)(value * 255)); 
					if (m_AutoRender) Render();
				}
			}

            internal Font m_LabelFont = new Font("Arial", 12);

            /// <summary>
            /// Font name to be used for labels.
            /// </summary>
            public string LabelFontName
            {
                get { return m_LabelFont.Name; }
                set
                {
                    m_LabelFont = new Font(value, m_LabelFont.Size);
                    if (m_AutoRender) Render();
                }
            }

            /// <summary>
            /// Font size to be used for labels.
            /// </summary>
            public int LabelFontSize
            {
                get { return (int)m_LabelFont.Size; }
                set
                {
                    m_LabelFont = new Font(m_LabelFont.Name, value);
                    if (m_AutoRender) Render();
                }
            }

            internal int m_LabelOffsetX = -4;
            internal int m_LabelOffsetY = -4;

            /// <summary>
            /// Sets the graphical offset in points between an object and its label.
            /// </summary>
            /// <param name="xoff">x offset.</param>
            /// <param name="yoff">y offset.</param>
            public void SetLabelOffset(int xoff, int yoff)
            {
                m_LabelOffsetX = xoff;
                m_LabelOffsetY = yoff;
                if (m_AutoRender) Render();
            }

            /// <summary>
            /// Sets the highlight status for all graphical objects associated to a specified owner.
            /// </summary>
            /// <param name="owner">the owner whose object should change status.</param>
            /// <param name="hlstatus"><c>true</c> to highlight, <c>false</c> to show in normal fashion</param>
            public void Highlight(object owner, bool hlstatus)
            {
                int i;
                for (i = 0; i < m_Points.Count; i++)
                {
                    Point p = ((Point)m_Points[i]);
                    if (p.Owner == owner) p.Highlight = hlstatus;
                }
                for (i = 0; i < m_Lines.Count; i++)
                {
                    Line l = ((Line)m_Lines[i]);
                    if (l.Owner == owner) l.Highlight = hlstatus;
                }
                if (m_AutoRender) Render();
            }

            /// <summary>
            /// Sets the label status for all graphical objects associated to a specified owner.
            /// </summary>
            /// <param name="owner">the owner whose label should change status.</param>
            /// <param name="hlstatus"><c>true</c> to show label, <c>false</c> to hide label.</param>
            public void EnableLabel(object owner, bool enable)
            {
                int i;
                for (i = 0; i < m_Points.Count; i++)
                {
                    Point p = ((Point)m_Points[i]);
                    if (p.Owner == owner) p.EnableLabel = enable;
                }
                for (i = 0; i < m_Lines.Count; i++)
                {
                    Line l = ((Line)m_Lines[i]);
                    if (l.Owner == owner) l.EnableLabel = enable;
                }
                if (m_AutoRender) Render();
            }

            /// <summary>
            /// Sets the label or all graphical objects associated to a specified owner.
            /// </summary>
            /// <param name="owner">the owner whose label should be set.</param>
            /// <param name="hlstatus">the new label text.</param>
            /// <remarks>If an object was created with its label = <c>null</c>, its label will never change. 
            /// In order to set an object with an empty label that can be set later, its label must be set to the empty string (<c>""</c>).
            /// </remarks>
            public void SetLabel(object owner, string newlabel)
            {
                if (newlabel == null) return;
                int i;
                for (i = 0; i < m_Points.Count; i++)
                {
                    Point p = ((Point)m_Points[i]);
                    if (p.Owner == owner && p.Label != null) p.Label = newlabel;
                }
                for (i = 0; i < m_Lines.Count; i++)
                {
                    Line l = ((Line)m_Lines[i]);
                    if (l.Owner == owner && l.Label != null) l.Label = newlabel;
                }
                if (m_AutoRender) Render();
            }

            /// <summary>
            /// Records the current image to a movie.
            /// </summary>
            /// <param name="mv">the movie to record the image to.</param>
            /// <returns>the number of frames in the movie.</returns>
            public int Record(Movie mv)
            {
                Render();
                return mv.AddFrame(BufferBitmap);
            }

			internal void Render()
			{
				if (m_Changed) Transform();			
				BufferG.FillRectangle(new SolidBrush(this.m_BackColor), 0, 0, BufferBitmap.Width, BufferBitmap.Height);
                foreach (Point p in m_Points)
                {
                    Brush thebrush = new SolidBrush(Color.FromArgb(m_Alpha, p.R, p.G, p.B));
                    int ps = m_PointSize + (p.Highlight ? 4 : 0);
                    BufferG.FillEllipse(thebrush, p.TX - ps / 2, p.TY - ps / 2, ps, ps);
                    if (p.EnableLabel && p.Label != null && p.Label.Length > 0) BufferG.DrawString(p.Label, m_LabelFont, thebrush, p.TX + m_LabelOffsetX, p.TY + m_LabelOffsetY);
                }
                foreach (Line l in m_Lines)
                {
                    BufferG.DrawLine(new Pen(Color.FromArgb(m_Alpha, l.R, l.G, l.B), m_LineWidth + (l.Highlight ? 2 : 0)), l.TXF, l.TYF, l.TXS, l.TYS);
                    if (l.EnableLabel && l.Label != null && l.Label.Length > 0) BufferG.DrawString(l.Label, m_LabelFont, new SolidBrush(Color.FromArgb(m_Alpha, l.R, l.G, l.B)), (l.TXS + l.TXF) / 2 + m_LabelOffsetX, (l.TYS + l.TYF) / 2 + m_LabelOffsetY);
                }
			}

            /// <summary>
            /// Removes all graphical objects owned by the specified owner.
            /// </summary>
            /// <param name="o">the owner whose objects have to be removed.</param>
            /// <returns>the number of elements deleted.</returns>
            public int RemoveOwned(object o)
            {
                int i;
                int n = 0;
                for (i = 0; i < m_Points.Count; i++)
                    if (((GDI3DObject)m_Points[i]).Owner == o)
                    {
                        m_Points.RemoveAt(i--);
                        n++;
                    }
                for (i = 0; i < m_Lines.Count; i++)
                    if (((GDI3DObject)m_Lines[i]).Owner == o)
                    {
                        m_Lines.RemoveAt(i--);
                        n++;
                    }
                if (m_AutoRender) Render();
                return n;
            }

            internal void Render(System.Drawing.Graphics g)
            {
                if (m_Changed) Transform();
                g.FillRectangle(new SolidBrush(this.m_BackColor), 0, 0, g.ClipBounds.Width, g.ClipBounds.Height);
                foreach (Point p in m_Points)
                {
                    Brush thebrush = new SolidBrush(Color.FromArgb(m_Alpha, p.R, p.G, p.B));
                    int ps = m_PointSize + (p.Highlight ? 4 : 0);
                    g.FillEllipse(thebrush, p.TX - ps / 2, p.TY - ps / 2, ps, ps);
                    if (p.EnableLabel && p.Label != null && p.Label.Length > 0) g.DrawString(p.Label, m_LabelFont, thebrush, p.TX + m_LabelOffsetX, p.TY + m_LabelOffsetY);
                }
                foreach (Line l in m_Lines)
                {
                    Pen p = new Pen(Color.FromArgb(m_Alpha, l.R, l.G, l.B), m_LineWidth + (l.Highlight ? 2 : 0));
                    if (l.Dashed)
                    {
                        p.DashStyle = System.Drawing.Drawing2D.DashStyle.Custom;
                        p.DashPattern = new float[2] { 4.0f, 4.0f };
                    }
                    g.DrawLine(p, l.TXF, l.TYF, l.TXS, l.TYS);                    
                    if (l.EnableLabel && l.Label != null && l.Label.Length > 0) g.DrawString(l.Label, m_LabelFont, new SolidBrush(Color.FromArgb(m_Alpha, l.R, l.G, l.B)), (l.TXS + l.TXS) / 2 + m_LabelOffsetX, (l.TYS + l.TYS) / 2 + m_LabelOffsetY);
                }
            }

            /// <summary>
			/// Transforms 3D coordinates for all objects to 2D coordinates for drawing.
			/// </summary>
			public void Transform()
			{
				int i;
				double zoom = Math.Min(m_Zoom * BufferBitmap.Width, m_Zoom * BufferBitmap.Height);
				int xc = BufferBitmap.Width / 2;
				int yc = BufferBitmap.Height / 2;
				for (i = 0; i < m_Points.Count; i++)
				{
					Point p = (Point)m_Points[i];
					double px = p.X - m_CameraPosX;
					double py = p.Y - m_CameraPosY;
					double pz = p.Z - m_CameraPosZ;
					double tz = px * m_CameraDirectionX + py * m_CameraDirectionY + pz * m_CameraDirectionZ;
					if (tz <= 0.0) 
					{
						p.Show = false;
						continue;
					}
					p.Show = true;				
					double tx = px * m_CameraBinormalX + py * m_CameraBinormalY + pz * m_CameraBinormalZ;
					double ty = px * m_CameraNormalX + py * m_CameraNormalY + pz * m_CameraNormalZ;
					if (m_Infinity)
					{
						p.TX = (int)(m_Zoom * tx + xc);
						p.TY = (int)(m_Zoom * ty + yc);
					}
					else
					{
						p.TX = (int)(m_Zoom * tx / tz + xc);
						p.TY = (int)(m_Zoom * ty / tz + yc);
					}
				}
				for (i = 0; i < m_Lines.Count; i++)
				{
					Line l = (Line)m_Lines[i];
					double lxs = l.XS - m_CameraPosX;
					double lys = l.YS - m_CameraPosY;
					double lzs = l.ZS - m_CameraPosZ;
					double lxf = l.XF - m_CameraPosX;
					double lyf = l.YF - m_CameraPosY;
					double lzf = l.ZF - m_CameraPosZ;
					double sz = lxs * m_CameraDirectionX + lys * m_CameraDirectionY + lzs * m_CameraDirectionZ;
					if (sz <= 0.0) 
					{
						l.Show = false;
						continue;
					}
					double fz = lxf * m_CameraDirectionX + lyf * m_CameraDirectionY + lzf * m_CameraDirectionZ;
					if (fz <= 0.0) 
					{
						l.Show = false;
						continue;
					}
					l.Show = true;
					double sx = lxs * m_CameraBinormalX + lys * m_CameraBinormalY + lzs * m_CameraBinormalZ;
					double sy = lxs * m_CameraNormalX + lys * m_CameraNormalY + lzs * m_CameraNormalZ;
					if (m_Infinity)
					{
						l.TXS = (int)(m_Zoom * sx + xc);
						l.TYS = (int)(m_Zoom * sy + yc);
					}
					else
					{
						l.TXS = (int)(m_Zoom * sx / sz + xc);
						l.TYS = (int)(m_Zoom * sy / sz + yc);
					}
					double fx = lxf * m_CameraBinormalX + lyf * m_CameraBinormalY + lzf * m_CameraBinormalZ;
					double fy = lxf * m_CameraNormalX + lyf * m_CameraNormalY + lzf * m_CameraNormalZ;
					if (m_Infinity)
					{
						l.TXF = (int)(m_Zoom * fx + xc);
						l.TYF = (int)(m_Zoom * fy + yc);
					}
					else
					{
						l.TXF = (int)(m_Zoom * fx / fz + xc);
						l.TYF = (int)(m_Zoom * fy / fz + yc);
					}
				}
				m_Changed = false;
			}

			internal System.Drawing.Bitmap BufferBitmap = null;

			internal System.Drawing.Graphics BufferG = null;

            /// <summary>
            /// Sets the scene.
            /// </summary>
            /// <param name="value">the new scene to be set.</param>
            public void SetScene(GDI3D.Scene value)
            {
                this.BackColor = Color.FromArgb(value.BackColor.R, value.BackColor.G, value.BackColor.B);
                this.m_CameraDirectionX = value.CameraDirectionX;
                this.m_CameraDirectionY = value.CameraDirectionY;
                this.m_CameraDirectionZ = value.CameraDirectionZ;
                this.m_CameraNormalX = value.CameraNormalX;
                this.m_CameraNormalY = value.CameraNormalY;
                this.m_CameraNormalZ = value.CameraNormalZ;
                this.m_CameraBinormalX = m_CameraDirectionY * m_CameraNormalZ - m_CameraDirectionZ * m_CameraNormalY;
                this.m_CameraBinormalY = m_CameraDirectionZ * m_CameraNormalX - m_CameraDirectionX * m_CameraNormalZ;
                this.m_CameraBinormalZ = m_CameraDirectionX * m_CameraNormalY - m_CameraDirectionY * m_CameraNormalX;
                this.m_CameraSpottingX = value.CameraSpottingX;
                this.m_CameraSpottingY = value.CameraSpottingY;
                this.m_CameraSpottingZ = value.CameraSpottingZ;
                this.m_Distance = value.CameraDistance;
                this.m_CameraPosX = m_CameraSpottingX - m_Distance * m_CameraDirectionX;
                this.m_CameraPosY = m_CameraSpottingY - m_Distance * m_CameraDirectionY;
                this.m_CameraPosZ = m_CameraSpottingZ - m_Distance * m_CameraDirectionZ;
                this.m_Zoom = value.Zoom;
                this.m_Changed = true;
                this.m_Points.Clear();
                this.m_Lines.Clear();
                foreach (GDI3D.Point gp in value.Points)
                    m_Points.Add(new Point(gp.X, gp.Y, gp.Z, (gp.Owner < 0) ? null : value.OwnerSignatures[gp.Owner], gp.R, gp.G, gp.B));
                foreach (GDI3D.Line gl in value.Lines)
                    m_Lines.Add(new Line(gl.XF, gl.YF, gl.ZF, gl.XS, gl.YS, gl.ZS, (gl.Owner < 0) ? null : value.OwnerSignatures[gl.Owner], gl.R, gl.G, gl.B));
                if (m_AutoRender) Render();
            }
            /// <summary>
            /// Finds the object that is nearest to a certain x,y position.
            /// </summary>
            /// <param name="pos_x">the x coordinate of the point being searched (in graphical coordinates).</param>
            /// <param name="pos_y">the y coordinate of the point being searched (in graphical coordinates).</param>
            /// <param name="centerwhenfound">if <c>true</c>, the plot is centered on the object selected (and left unchanged if no selectable object is found).</param>
            /// <returns>if an object is found, its string representation is returned, <c>null</c> otherwise.</returns>
            public string FindNearestObject(int pos_x, int pos_y, bool centerwhenfound)
            {
                object o = null;
                double d = 0.0, bestd = 0.0;
                double x = 0.0, y = 0.0, z = 0.0;
                foreach (Point p in m_Points)
                {
                    if (p.Owner == null) continue;
                    d = p.Distance2(pos_x, pos_y);
                    if (o == null || d < bestd)
                    {
                        bestd = d;
                        o = p;
                        x = p.X;
                        y = p.Y;
                        z = p.Z;
                    }
                }
                foreach (Line l in m_Lines)
                {
                    if (l.Owner == null) continue;
                    d = l.Distance2(pos_x, pos_y);
                    if (o == null || d < bestd)
                    {
                        bestd = d;
                        o = l;
                        x = (l.XF + l.XS) * 0.5;
                        y = (l.YF + l.YS) * 0.5;
                        z = (l.ZF + l.ZS) * 0.5;
                    }
                }
                if (o == null) return null;
                if (centerwhenfound)
                {
                    SetCameraSpotting(x, y, z);
                }
                if (o is GDI3D.Plot.Line) return ((GDI3D.Plot.Line)o).Owner.ToString();
                if (o is GDI3D.Plot.Point) return ((GDI3D.Plot.Point)o).Owner.ToString();
                return o.GetType().ToString();
            }

        }

        /// <summary>
        /// A generic graphical object.
        /// </summary>        
        public class GDI3DObject
        {
            /// <summary>
            /// Owner object.
            /// </summary>
            public object Owner;
            /// <summary>
            /// Label for the graphical object.
            /// </summary>
            public string Label;
            /// <summary>
            /// <c>true</c> if the label is to be shown, <c>false</c> otherwise.
            /// </summary>
            public bool EnableLabel;
            /// <summary>
            /// <c>true</c> if the object is to be highlighted, <c>false</c> otherwise;
            /// </summary>
            public bool Highlight;
        }

		/// <summary>
		/// A display point.
		/// </summary>
		public class Point : GDI3DObject
		{
			/// <summary>
			/// X component.
			/// </summary>
			public double X;
			/// <summary>
			/// Y component.
			/// </summary>
			public double Y;
			/// <summary>
			/// Z component.
			/// </summary>
			public double Z;
			/// <summary>
			/// Red component.
			/// </summary>
			public int R;
			/// <summary>
			/// Green component.
			/// </summary>
			public int G;
			/// <summary>
			/// Blue component.
			/// </summary>
			public int B;
			/// <summary>
			/// Transformed X.
			/// </summary>
			public int TX;
			/// <summary>
			/// Transformed Y;
			/// </summary>
			public int TY;
			/// <summary>
			/// Set if the line can be shown (i.e. it is inside viewport).
			/// </summary>
			public bool Show;
			/// <summary>
			/// Default Red component.
			/// </summary>
			public static int DefaultR = 255;
			/// <summary>
			/// Default Green component.
			/// </summary>
			public static int DefaultG = 255;
			/// <summary>
			/// Default Blue component.
			/// </summary>
			public static int DefaultB = 255;
			/// <summary>
			/// Constructs a new white point.
			/// </summary>
			/// <param name="x">X component.</param>
			/// <param name="y">Y component.</param>
			/// <param name="z">Z component.</param>
			/// <param name="owner">Owner object.</param>
			public Point(double x, double y, double z, object owner)
			{
				X = x;
				Y = y;
				Z = z;
				Owner = owner;
				R = DefaultR;
				G = DefaultG;
				B = DefaultB;
				TX = TY = 0;
				Show = false;
			}
			/// <summary>
			/// Constructs a colored point.
			/// </summary>
			/// <param name="x">X component.</param>
			/// <param name="y">Y component.</param>
			/// <param name="z">Z component.</param>
			/// <param name="owner">Owner object.</param>
			/// <param name="r">Red component.</param>
			/// <param name="g">Green component.</param>
			/// <param name="b">Blue component.</param>
			public Point(double x, double y, double z, object owner, int r, int g, int b)
			{
				X = x;
				Y = y;
				Z = z;
				Owner = owner;
				R = r;
				G = g;
				B = b;
				TX = TY = 0;
				Show = false;
			}
			/// <summary>
			/// Distance-squared from a graphical point.
			/// </summary>
			/// <param name="XD">X component of point from which to distance is to be computed.</param>
			/// <param name="YD">Y component of point from which to distance is to be computed.</param>
			/// <returns></returns>
            public double Distance2(int XD, int YD)
            {
                double DX = TX - XD;
                double DY = TY - YD;
                return DX * DX + DY * DY;
            }
        }
		/// <summary>
		/// A display line.
		/// </summary>
		public class Line : GDI3DObject
		{
            /// <summary>
            /// <c>true</c> for dashed lines, <c>false</c> otherwise;
            /// </summary>
            public bool Dashed;
			/// <summary>
			/// First point X component;
			/// </summary>
			public double XF;
			/// <summary>
			/// First point Y component;
			/// </summary>
			public double YF;
			/// <summary>
			/// First point Z component;
			/// </summary>
			public double ZF;
			/// <summary>
			/// Second point X component;
			/// </summary>
			public double XS;
			/// <summary>
			/// Second point Y component;
			/// </summary>
			public double YS;
			/// <summary>
			/// Second point Z component;
			/// </summary>
			public double ZS;
			/// <summary>
			/// Red component;
			/// </summary>
			public int R;
			/// <summary>
			/// Green component;
			/// </summary>
			public int G;
			/// <summary>
			/// Blue component;
			/// </summary>
			public int B;
			/// <summary>
			/// Transformed X component of first point.
			/// </summary>
			public int TXF;
			/// <summary>
			/// Transformed Y component of first point.
			/// </summary>
			public int TYF;
			/// <summary>
			/// Transformed X component of second point.
			/// </summary>
			public int TXS;
			/// <summary>
			/// Transformed Y component of second point.
			/// </summary>
			public int TYS;
			/// <summary>
			/// Set if the line can be shown (i.e. it is inside viewport).
			/// </summary>
			public bool Show;
			/// <summary>
			/// Default Red component.
			/// </summary>
			public static int DefaultR = 255;
			/// <summary>
			/// Default Green component.
			/// </summary>
			public static int DefaultG = 255;
			/// <summary>
			/// Default Blue component.
			/// </summary>
			public static int DefaultB = 255;
			/// <summary>
			/// Constructs a line with default color.
			/// </summary>
			/// <param name="xf">X component of first point.</param>
			/// <param name="yf">Y component of first point.</param>
			/// <param name="zf">Z component of first point.</param>
			/// <param name="xs">X component of second point.</param>
			/// <param name="ys">Y component of second point.</param>
			/// <param name="zs">Z component of second point.</param>
			/// <param name="owner">Owner object.</param>
			public Line(double xf, double yf, double zf, double xs, double ys, double zs, object owner)
			{
				XS = xs;
				YS = ys;
				ZS = zs;
				XF = xf;
				YF = yf;
				ZF = zf;
				Owner = owner;
				R = DefaultR;
				G = DefaultG;
				B = DefaultB;
				TXS = TYS = TXF = TYF = 0;
				Show = false;
			}
			/// <summary>
			/// Constructs a colored line.
			/// </summary>
			/// <param name="xf">X component of first point.</param>
			/// <param name="yf">Y component of first point.</param>
			/// <param name="zf">Z component of first point.</param>
			/// <param name="xs">X component of second point.</param>
			/// <param name="ys">Y component of second point.</param>
			/// <param name="zs">Z component of second point.</param>
			/// <param name="owner">Owner object.</param>
			/// <param name="r">Red component.</param>
			/// <param name="g">Green component.</param>
			/// <param name="b">Blue component.</param>
			public Line(double xf, double yf, double zf, double xs, double ys, double zs, object owner, int r, int g, int b)
			{
				XS = xs;
				YS = ys;
				ZS = zs;
				XF = xf;
				YF = yf;
				ZF = zf;
				Owner = owner;
				R = r;
				G = g;
				B = b;			
				TXS = TYS = TXF = TYF = 0;
				Show = false;
			}
			/// <summary>
			/// Distance-squared from a graphical point.
			/// </summary>
			/// <param name="XD">X component of point from which to distance is to be computed.</param>
			/// <param name="YD">Y component of point from which to distance is to be computed.</param>
			/// <returns></returns>
            public double Distance2(int XD, int YD)
            {
                double NX = TXS - TXF;
                double NY = TYS - TYF;
                double DXS = TXS - XD;
                double DYS = TYS - YD;
                if (NX == 0.0 && NY == 0.0) return DXS * DXS + DYS * DYS;
                double n2 = 1.0 / (NX * NX + NY * NY);
                double DXF = XD - TXF;
                double DYF = YD - TYF;
                double p = (DXF * NX + DYF * NY) * n2;
                if (p < 0.0) return DXF * DXF + DYF * DYF;
                if (p > 1.0) return DXS * DXS + DYS * DYS;
                p = DXF * NY - DYF * NX;
                return p * p * n2;
            }
        }
	}
}
