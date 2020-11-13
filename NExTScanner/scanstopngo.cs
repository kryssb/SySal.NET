private void ScanStopNGo()
        {            
            System.Collections.ArrayList traj_arr = new System.Collections.ArrayList();
            TerminateSignal.Reset();
            int igpu;
            try
            {                
                iCamDisp.EnableAutoRefresh = false;
                iGrab.SequenceSize = (int)WorkConfig.Layers;
                double fovwidth = Math.Abs(ImagingConfig.ImageWidth * ImagingConfig.Pixel2Micron.X);
                double fovheight = Math.Abs(ImagingConfig.ImageHeight * ImagingConfig.Pixel2Micron.Y);
                double stepwidth = fovwidth - WorkConfig.ViewOverlap;
                double stepheight = fovheight - WorkConfig.ViewOverlap;
                double stepxspeed = WorkConfig.ContinuousMotionDutyFraction * stepwidth * WorkConfig.FramesPerSecond / WorkConfig.Layers;
                double stepyspeed = WorkConfig.ContinuousMotionDutyFraction * stepheight * WorkConfig.FramesPerSecond / WorkConfig.Layers;
                double lowestz = iStage.GetNamedReferencePosition("LowestZ");
                double[] expectedzcenters = new double[]
                {
                    lowestz + WorkConfig.EmulsionThickness * 1.5 + WorkConfig.BaseThickness, lowestz + WorkConfig.EmulsionThickness * 0.5
                };
                SyncZData[] ZData = new SyncZData[] { new SyncZData(), new SyncZData() };
                ZData[0].WriteZData(false, expectedzcenters[0]);
                ZData[1].WriteZData(false, expectedzcenters[1]);
                if (stepheight <= 0.0 || stepwidth <= 0.0) throw new Exception("Too much overlap, or null image or pixel/micron factor defined.");
                SySal.BasicTypes.Vector2 StripDelta = new BasicTypes.Vector2();
                SySal.BasicTypes.Vector2 ViewDelta = new BasicTypes.Vector2();
                SySal.BasicTypes.Vector ImageDelta = new BasicTypes.Vector();
                ImageDelta.Z = -WorkConfig.Pitch;
                int views, strips;
                if (WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X)
                {
                    ImageDelta.Y = 0.0;
                    ImageDelta.X = (WorkConfig.Layers == 0) ? 0.0 : (WorkConfig.ContinuousMotionDutyFraction * stepwidth / (WorkConfig.Layers - 1));
                    ViewDelta.X = stepwidth;
                    ViewDelta.Y = 0.0;
                    views = Math.Max(1, (int)Math.Ceiling((ScanRectangle.MaxX - ScanRectangle.MinX) / stepwidth));
                    StripDelta.X = 0.0;
                    StripDelta.Y = stepheight;
                    strips = Math.Max(1, (int)Math.Ceiling((ScanRectangle.MaxY - ScanRectangle.MinY) / stepheight));
                }
                else
                {
                    ImageDelta.X = 0.0;
                    ImageDelta.Y = (WorkConfig.Layers == 0) ? 0.0 : (WorkConfig.ContinuousMotionDutyFraction * stepheight / (WorkConfig.Layers - 1));
                    ViewDelta.X = 0.0;
                    ViewDelta.Y = stepheight;
                    views = Math.Max(1, (int)Math.Ceiling((ScanRectangle.MaxY - ScanRectangle.MinY) / stepheight));
                    StripDelta.X = stepwidth;
                    StripDelta.Y = 0.0;
                    strips = Math.Max(1, (int)Math.Ceiling((ScanRectangle.MaxX - ScanRectangle.MinX) / stepwidth));
                }
                int i_strip, i_view, i_side, i_image;
                int firstlayer, lastlayer;
                int totalviews = strips * views * ((WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both) ? 2 : 1);
                SetProgressValue(0);
                SetProgressMax(totalviews);
                SetQueueLengthMax(views);                
                int viewsok = 0;
                int[] clustercounts = new int[WorkConfig.Layers];
                GrabReadySignal = new int[views];
                ProcReadySignal = new bool[views];
                GrabDataSlots = new GrabData[views];
                ProcDataSlots = new ProcOutputData[views];
                TerminateSignal.Reset();
                SetQueueLength();
                if (false)
                {
                    /* GPU TESTING */
                    int seqsize;
                    for (seqsize = 1; seqsize <= WorkConfig.Layers; seqsize++)
                    {
                        iLog.Log("GPU TESTING", "Step 0 SeqSize " + seqsize);
                        iGrab.SequenceSize = seqsize;
                        iLog.Log("GPU TESTING", "Step 1");
                        object test_gseq = iGrab.GrabSequence();
                        iLog.Log("GPU TESTING", "Step 2");
                        SySal.Imaging.LinearMemoryImage test_lmi = (SySal.Imaging.LinearMemoryImage)iGrab.MapSequenceToSingleImage(test_gseq);
                        iLog.Log("GPU TESTING", "Step 3");
                        iGrab.ClearGrabSequence(test_gseq);
                        iLog.Log("GPU TESTING", "Step 4");
                        for (igpu = 0; igpu < iGPU.Length; igpu++)
                        {
                            iLog.Log("GPU TESTING", "Step 5 - GPU " + igpu);
                            iGPU[igpu].Input = test_lmi;
                            iLog.Log("GPU TESTING", "Step 6 - GPU " + igpu);
                        }
                        iLog.Log("GPU TESTING", "Step 7");
                        iGrab.ClearMappedImage(test_lmi);
                        iLog.Log("GPU TESTING", "Step 8 SeqSize " + seqsize);
                    }
                    iGrab.SequenceSize = (int)WorkConfig.Layers;
                }
                ImgProcThreads = new System.Threading.Thread[iGPU.Length];
                for (igpu = 0; igpu < iGPU.Length; igpu++)
                    (ImgProcThreads[igpu] = new System.Threading.Thread(new System.Threading.ParameterizedThreadStart(ImageProcessingThread))).Start(igpu);
                System.DateTime starttime = System.DateTime.Now;
                for (i_strip = 0; i_strip < strips && ShouldStop == false; i_strip++)                                    
                    for (i_side = (((WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both) || (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Top)) ? 0 : 1);
                        i_side <= (((WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both) || (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Bottom)) ? 1 : 0) && ShouldStop == false;
                        i_side++)
                        {
                            for (i_view = 0; i_view < views; i_view++)
                            {
                                GrabReadySignal[i_view] = 0;
                                ProcReadySignal[i_view] = false;
                                GrabDataSlots[i_view] = null;
                                ProcDataSlots[i_view] = null;
                            }
                            iLog.Log("Scan", "Side " + i_side + " WorkConfig.Sides " + WorkConfig.Sides + " " + (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both) + " " + (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Top) + " " + (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Bottom));
                            QuasiStaticAcquisition qa = new QuasiStaticAcquisition();
                            qa.FilePattern = QuasiStaticAcquisition.GetFilePattern(m_QAPattern, i_side == 0, (uint)i_strip);
                            qa.Sequences = new QuasiStaticAcquisition.Sequence[views];
                            for (i_view = 0; i_view < views && ShouldStop == false; i_view++)
                            {
                                QuasiStaticAcquisition.Sequence seq = qa.Sequences[i_view] = new QuasiStaticAcquisition.Sequence();
                                seq.Owner = qa;
                                seq.Id = (uint)i_view;
                                seq.Layers = new QuasiStaticAcquisition.Sequence.Layer[WorkConfig.Layers];
                                SetProgressValue(viewsok);
                                for (i_image = 0; i_image < clustercounts.Length; i_image++) clustercounts[i_image] = -1;
                                firstlayer = (int)WorkConfig.Layers;
                                lastlayer = -1;
                                double tx = ScanRectangle.MinX + ViewDelta.X * i_view + StripDelta.X * i_strip;
                                double ty = ScanRectangle.MinY + ViewDelta.Y * i_view + StripDelta.Y * i_strip;
                                bool ok = false;
                                double tz = 0.0;
                                ZData[i_side].ReadZData(ref ok, ref tz);
                                if (ok == false)
                                {
                                    if (GoToPos(tx, ty, expectedzcenters[i_side], false) == false) throw new Exception("Cannot reach sweep start position at strip " + i_strip + ", view " + i_view + " side " + i_side + ".");
                                    double topz = 0.0, bottomz = 0.0;
                                    ok = FindEmulsionZs(expectedzcenters[i_side], ref topz, ref bottomz);
                                    if (ok == false)
                                    {
                                        /* make empty view */
                                        continue;
                                    }
                                    tz = 0.5 * (topz + bottomz);
                                    ZData[i_side].WriteZData(ok,tz); 
                                    SetStatus("FindEmulsionZs: " + ok + " top " + topz + " bottom " + bottomz + " thickness " + (topz - bottomz));
                                }
                                tz += WorkConfig.ZSweep * 0.5;
                                iStage.StartRecording(1.0, WorkConfig.Layers / WorkConfig.FramesPerSecond * 1000.0);
                                if (GoToPos(tx, ty, tz + WorkConfig.PositionTolerance, true) == false) throw new Exception("Cannot reach sweep start position at strip " + i_strip + ", view " + i_view + " side " + i_side + ".");
                                traj_arr.AddRange(iStage.Trajectory);
                                tx += ImageDelta.X * (WorkConfig.Layers - 1);
                                ty += ImageDelta.Y * (WorkConfig.Layers - 1);
                                iStage.PosMove(SySal.StageControl.Axis.X, tx, stepxspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                iStage.PosMove(SySal.StageControl.Axis.Y, ty, stepyspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                iStage.PosMove(SySal.StageControl.Axis.Z, tz - WorkConfig.ZSweep, WorkConfig.ZSweepSpeed, WorkConfig.ZAcceleration, WorkConfig.ZAcceleration);
                                bool axiserror = false;
                                while (ShouldStop == false && iStage.GetPos(StageControl.Axis.Z) > tz && (axiserror = (iStage.GetStatus(StageControl.Axis.Z) != StageControl.AxisStatus.OK)));
                                iStage.StartRecording(1.0, WorkConfig.Layers / WorkConfig.FramesPerSecond * 1000.0);
                                GrabData gd = new GrabData();
                                gd.strip = i_strip;
                                gd.side = i_side;
                                gd.view = i_view;
                                gd.GrabSeq = iGrab.GrabSequence();
                                gd.TimeInfo = iGrab.GetImageTimesMS(gd.GrabSeq);
                                gd.StageInfo = iStage.Trajectory;
                                GrabDataSlots[i_view] = gd;
                                GrabReadySignal[i_view] = -1;
                                traj_arr.AddRange(gd.StageInfo);
                                viewsok++;
                                SetQueueLength();
                            }
                            if (ShouldStop == false)
                            {
                                for (i_view = 0; i_view < views; i_view++)
                                {
                                    while (ProcReadySignal[i_view] == false)
                                        System.Threading.Thread.Yield();
                                    ProcOutputData po = ProcDataSlots[i_view];
                                    if (po != null)
                                    {
                                        qa.Sequences[i_view] = new QuasiStaticAcquisition.Sequence();
                                        qa.Sequences[i_view].Id = (uint)i_view;
                                        qa.Sequences[i_view].Owner = qa;
                                        qa.Sequences[i_view].Layers = new QuasiStaticAcquisition.Sequence.Layer[po.ImagePositionInfo.Length];
                                        for (i_image = 0; i_image < po.ImagePositionInfo.Length; i_image++)
                                        {
                                            qa.Sequences[i_view].Layers[i_image] = new QuasiStaticAcquisition.Sequence.Layer();
                                            qa.Sequences[i_view].Layers[i_image].Owner = qa.Sequences[i_view];
                                            qa.Sequences[i_view].Layers[i_image].Id = (uint)i_image;
                                            qa.Sequences[i_view].Layers[i_image].Clusters = (uint)po.Clusters.ClustersInImage(i_image);
                                            qa.Sequences[i_view].Layers[i_image].Position = po.ImagePositionInfo[i_image].Position;
                                            qa.Sequences[i_view].Layers[i_image].WriteSummary();
                                        }
                                    }
                                }
                            }                        
                    }
                SetProgressValue(viewsok);
                TerminateSignal.Set();
                for (igpu = 0; igpu < iGPU.Length; igpu++)
                    ImgProcThreads[igpu].Join();
                ImgProcThreads = new System.Threading.Thread[0];
                System.DateTime endtime = System.DateTime.Now;
                iLog.Log("Scan", "Total time: " + (endtime - starttime));
            }
            catch (Exception xc)
            {
                iLog.Log("Scan error", xc.ToString());
            }
            finally
            {
                try
                {
                    TerminateSignal.Set();
                    for (igpu = 0; igpu < ImgProcThreads.Length; igpu++)
                        ImgProcThreads[igpu].Join();
                    ImgProcThreads = new System.Threading.Thread[0];

                    SetStatus("Dumping trajectory");
                    string tswfile = DataDir;
                    if (tswfile.EndsWith("\\") == false && tswfile.EndsWith("/") == false) tswfile += "/";
                    tswfile += "debug_trajdump.txt";
                    System.IO.StreamWriter tsw = new System.IO.StreamWriter(tswfile);
                    tsw.WriteLine("ID\tT\tX\tY\tZ");
                    int i;
                    for (i = 0; i < traj_arr.Count; i++)
                    {
                        SySal.StageControl.TrajectorySample ts = (SySal.StageControl.TrajectorySample)traj_arr[i];
                        tsw.WriteLine(i + "\t" + ts.TimeMS + "\t" + ts.Position.X + "\t" + ts.Position.Y + "\t" + ts.Position.Z);
                    }
                    tsw.Flush();
                    tsw.Close();
                }
                catch (Exception xcx)
                {
                    MessageBox.Show("Scan", xcx.ToString());
                }
                SetStatus("Done");
                ShouldStop = true;
                EnableControls(true);
                iGrab.SequenceSize = 1;
                iCamDisp.EnableAutoRefresh = true;
            }
        }