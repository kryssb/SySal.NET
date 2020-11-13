using System;
using System.IO;
using System.Collections.Generic;
using System.Text;

namespace SySal.Executables.SecondaryVertexAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("");
            if (args.Length == 11)
                Console.WriteLine("id vertex coordinate from input data");
            else if (args.Length == 14)
                Console.WriteLine("vertex coordinate from external data");
            else
            {
                Console.WriteLine("---------- ERROR! ----------");
                Console.WriteLine("");
                Console.WriteLine("the program needs 11 or 14 parameters...you inserted " + args.Length + " parameters");
                Console.WriteLine("");
                Console.WriteLine("usage:");
                Console.WriteLine("SecondaryVertexAnalysis.exe <data type> <input file name> <output file name> <vertex id> <slope tolerance> <segments cut> <max plate> <max distance> <min Z> <start sx> <start sy>");
                Console.WriteLine("");
                Console.WriteLine("or");
                Console.WriteLine("");
                Console.WriteLine("SecondaryVertexAnalysis.exe <data type> <input file name> <output file name> -1 <slope tolerance> <segments cut> <max plate> <max distance> <min Z> <start sx> <start sy> <start x point> <start y point> <start z point>");
                Console.WriteLine("");
                Console.WriteLine("Supported data types");
                Console.WriteLine("tsr_v -> Load data from SySal TSR volume format.");
                Console.WriteLine("ascii_s -> Load data from ASCII n-tuple file.");
                Console.WriteLine("");
                Console.WriteLine("ASCII n-tuple files must have the format below:");
                Console.WriteLine("n_grains id_BT X Y SX SY Z");
                Console.WriteLine("n_grains -> number of grains of the segment(base track/microtrack).");
                Console.WriteLine("id_BT -> unique identifier of the segment(base track/microtrack).");
                Console.WriteLine("X Y SX SY Z -> obvious.");
                Console.WriteLine("");
                Console.WriteLine("The output is an ASCII n-tuple file using the format below:");
                Console.WriteLine("id_BT X Y Z SX Y");
                Console.WriteLine("----------------------------");
                Console.WriteLine("");
                Console.WriteLine("EXIT PROGRAM!!!");
                Console.WriteLine("");
                Console.Beep(200, 100);
                return;
            }
            Console.WriteLine("");
            //legge i parametri di input
            bool Program_check = true;

            int primary_vertex_id = Convert.ToInt32(args[3]);
            double slope_tolerance = Convert.ToDouble(args[4]);
            int N_seg = Convert.ToInt32(args[5]);
            int N_plate = Convert.ToInt32(args[6]);
            double distance_cut = Convert.ToDouble(args[7]);
            double Zmin = Convert.ToDouble(args[8]);



            double StartSX = Convert.ToDouble(args[9]);
            double StartSY = Convert.ToDouble(args[10]);

            Versor FirstVertex = new Versor();
            if (args.Length > 11)
                FirstVertex = Fill_versor(Convert.ToDouble(args[11]), Convert.ToDouble(args[12]), Convert.ToDouble(args[13]));
            else
                FirstVertex = Fill_versor(0, 0, 0);

            //controlla i parametri di input
            //--------------------------------------------------------------------------------
            if (slope_tolerance <= 0 || slope_tolerance >= 0.3)
            {
                Console.WriteLine("---------- ERROR at parameter 5 ----------");
                Console.WriteLine("The slope_tolerance must be > 0 and < 0.3");
                Console.WriteLine("----------------------------");
                Console.Beep(200, 100);
                Program_check = false;
            }
            if (N_seg < 1)
            {
                Console.WriteLine("---------- WARNING at parameter 6 ----------");
                Console.WriteLine("The minimum number of segments must be > 0");
                Console.WriteLine("------------------------------");
                Console.Beep(200, 100);
                N_seg = 1;
                Console.WriteLine("The number of segments is set on 1");
                Console.WriteLine("------------------------------");
            }
            if (N_plate < 1)
            {
                Console.WriteLine("---------- WARNING at parameter 7 ----------");
                Console.WriteLine("The minimum number of jumped plate must be > 0");
                Console.WriteLine("------------------------------");
                Console.Beep(200, 100);
                N_plate = 1;
                Console.WriteLine("The minimum number of jumped plate is set on 1");
                Console.WriteLine("------------------------------");
            }
            if (distance_cut < 0)
            {
                Console.WriteLine("---------- ERROR at parameter 8 ----------");
                Console.WriteLine("distance_cut must be > 0");
                Console.WriteLine("----------------------------");
                Console.Beep(200, 100);
                Program_check = false;
            }
            if (StartSX > 0.6)
            {
                Console.WriteLine("---------- ERROR at parameter 10 ----------");
                Console.WriteLine("the starting slope x can't be > 0.6");
                Console.WriteLine("----------------------------");
                Console.Beep(200, 100);
                Program_check = false;
            }
            if (StartSY > 0.6)
            {
                Console.WriteLine("---------- ERROR at parameter 11 ----------");
                Console.WriteLine("the starting slope y can't be > 0.6");
                Console.WriteLine("----------------------------");
                Console.Beep(200, 100);
                Program_check = false;
            }

            if (args[0] != "ascii_s" && args[0] != "tsr_v")
            {
                Console.WriteLine("---------- ERROR at parameter 1 ----------");
                Console.WriteLine("data type can be only: ascii_s or tsr_v");
                Console.WriteLine("----------------------------");
                Console.Beep(200, 100);
                Program_check = false;
            }

            if (!File.Exists(args[1]))
            {
                Console.WriteLine("---------- ERROR at parameter 2----------");
                Console.WriteLine("file don't exists!");
                Console.WriteLine("----------------------------");
                Program_check = false;
            }
            if (args[0] == "ascii_s" && args.Length != 14)
            {
                Console.WriteLine("---------- ERROR! ----------");
                Console.WriteLine("for data type ascii_s the program needs external coordinates");
                Console.WriteLine("----------------------------");
                Console.Beep(200, 100);
                Program_check = false;
            }
            if (args[0] == "ascii_s" && primary_vertex_id != -1)
            {
                Console.WriteLine("---------- ERROR at parameters 1 and 4 ----------");
                Console.WriteLine("for data type ascii_s the program needs external coordinates and primary_vertex_id must be -1");
                Console.WriteLine("----------------------------");
                Console.Beep(200, 100);
                Program_check = false;
            }
            if (primary_vertex_id == -1 && args.Length != 14)
            {
                Console.WriteLine("---------- ERROR! ----------");
                Console.WriteLine("THE PRIMARY_VERTEX_Id IS -1...so the program needs external coordinates");
                Console.WriteLine("----------------------------");
                Console.Beep(200, 100);
                Program_check = false;
            }
            if (args[0] == "tsr_v" && primary_vertex_id >= 0 && args.Length > 11)
            {
                Console.WriteLine("---------- ERROR! ----------");
                Console.WriteLine("THE PRIMARY_VERTEX_Id IS 1...so the program doesn't need external coordinates...what do you want to do?");
                Console.WriteLine("----------------------------");
                Console.Beep(200, 100);
                Program_check = false;
            }
            Console.WriteLine("");
            if (Program_check == false)
            {
                Console.WriteLine("EXIT PROGRAM!!!");
                Console.WriteLine("");
                return;
            }
            //--------------------------------------------------------------------------------


            //sens_to_cooplanarity = Math.Cos(Math.PI / 2 - sens_to_cooplanarity);
            double sens_to_cooplanarity = 0.05;

            //lancia l'algoritmo in sono contenuti gli algoritmi di ricerca tracce e vertici
            //--------------------------------------------------------------------------------
            Launch_Algorithms(args[0], args[1], args[2], primary_vertex_id, true, slope_tolerance, sens_to_cooplanarity, N_seg, distance_cut, N_plate, FirstVertex, Zmin, StartSX, StartSY);
            //--------------------------------------------------------------------------------

        }

        //memoria per l'output-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        static Info_date[] info_t_v = new Info_date[0];
        //---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        //funzione gestione lancio algoritmo e selezione IO--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        static void Launch_Algorithms(string File_type, string File_name, string Out_File_Name, int index, bool Vertex_Or_Track, double slope_tolerance, double sens_to_cooplanarity, int N_seg, double distance_cut, int N_plate, Versor FirstVertex, double Zmin, double Start_SX, double Start_SY)
        {

            Internal_I_Data I_Volume = new Internal_I_Data();

            if (File_type == "tsr_v")
            {

                SySal.TotalScan.Volume v = (SySal.TotalScan.Volume)SySal.OperaPersistence.Restore(File_name, typeof(SySal.TotalScan.Volume));
                Console.WriteLine(" Layers: " + v.Layers.Length + ", Tracks: " + v.Tracks.Length + ", Vertices: " + v.Vertices.Length);
                if (index > v.Vertices.Length)
                {
                    Console.WriteLine("VOLUME DOES NOT CONTAIN THE " + index + " VERTEX ID!");
                    Console.Beep(200, 100);
                    return;
                }
                if (index < 0)
                {
                    Console.WriteLine("Starting process from external coordinates");
                    Console.Beep(3000, 50);
                    Console.Beep(2000, 50);
                    Console.Beep(3000, 50);

                }
                else
                {
                    FirstVertex.x = v.Vertices[index].X;
                    FirstVertex.y = v.Vertices[index].Y;
                    FirstVertex.z = v.Vertices[index].Z;
                }
                I_Volume = TSR_I(v, FirstVertex, Zmin, Start_SX, Start_SY);

            }

            else if (File_type == "ascii_s")
            {
                I_Volume = Ascii_s(File_name, FirstVertex.z);

            }

            else
            {
                Console.WriteLine("input data type: " + File_type + " not implemented yet");
                return;
            }


            Console.WriteLine();
            Console.WriteLine("track and vertex search algorithms are working...please wait...");
            Console.WriteLine();
            if (I_Volume.I_tracks.Length > 0)
                tracks_search(I_Volume, index, 0, 0, FirstVertex, true, slope_tolerance, sens_to_cooplanarity, N_seg, distance_cut, N_plate, 0, Zmin, Start_SX, Start_SY);

            else Console.WriteLine("no tracks!");

            if (File_type == "tsr_v")
            {
                SySal.TotalScan.Volume v = (SySal.TotalScan.Volume)SySal.OperaPersistence.Restore(File_name, typeof(SySal.TotalScan.Volume));
                IO_data_TSR(v, Out_File_Name);
            }

            else if (File_type == "ascii_s")
            {
                IO_data_Ascii_s(I_Volume, Out_File_Name);
            }

        }

        //---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        //selettore tracce fake 
        static bool Tfake_search(Internal_I_Data tracks_volume_analysis, Int64 index_1, Int64 index_2, double slope, double Ip, double sx, double sy, Versor InTrack)
        {
            Int32 i;

            Info_date[] Indexes = new Info_date[0];

            bool flag = true;

            //double sx = tracks_volume_analysis.I_tracks[index_1].I_segments[tracks_volume_analysis.I_tracks[index_1].lenght - 1].sx;
            //double sy = tracks_volume_analysis.I_tracks[index_1].I_segments[tracks_volume_analysis.I_tracks[index_1].lenght - 1].sy;
            //double vertex_X = tracks_volume_analysis.I_tracks[index_1].I_segments[tracks_volume_analysis.I_tracks[index_1].lenght - 1].x;
            //double vertex_Y = tracks_volume_analysis.I_tracks[index_1].I_segments[tracks_volume_analysis.I_tracks[index_1].lenght - 1].y;
            //double vertex_Z = tracks_volume_analysis.I_tracks[index_1].I_segments[tracks_volume_analysis.I_tracks[index_1].lenght - 1].z;

            double vertex_X = InTrack.x;
            double vertex_Y = InTrack.y;
            double vertex_Z = InTrack.z;
            double slope_tolerance;
            double SlopeVertexVsTrakX = 0;
            double SlopeVertexVsTrakY = 0;
            double SlopeVertexVsTrakX2 = 0;
            double SlopeVertexVsTrakY2 = 0;

            double Distance = 0;

            Versor Zero_TTrack;

            slope_tolerance = 0.2;

            for (i = 0; i < tracks_volume_analysis.I_tracks.Length; i++)
            {


                flag = false;

                Internal_Track tk = tracks_volume_analysis.I_tracks[i];


                if (Search_info_dates(tk.id, tk.IpgId, false) == false && tk.I_segments[tk.lenght - 1].z < vertex_Z && tk.I_segments[tk.lenght - 1].z > (vertex_Z - 1500))
                {

                    Zero_TTrack = Fill_versor(vertex_X - tk.I_segments[0].x, vertex_Y - tk.I_segments[0].y, vertex_Z - tk.I_segments[0].z);

                    Distance = Module_Calculation(Zero_TTrack);

                    SlopeVertexVsTrakX = Zero_TTrack.x / Zero_TTrack.z;
                    SlopeVertexVsTrakY = Zero_TTrack.y / Zero_TTrack.z;

                    SlopeVertexVsTrakX2 = sx;
                    SlopeVertexVsTrakY2 = sy;
                    double Dx;
                    Dx = Sigma_calculation(vertex_X - (tk.I_segments[0].sx * (vertex_Z - tk.I_segments[0].z) + tk.I_segments[0].x), vertex_Y - (tk.I_segments[0].sy * (vertex_Z - tk.I_segments[0].z) + tk.I_segments[0].y));
                    Distance = Dx;

                    if ((Sigma_calculation(SlopeVertexVsTrakX - SlopeVertexVsTrakX2, SlopeVertexVsTrakY - SlopeVertexVsTrakY2) <= slope_tolerance))
                    {
                        flag = true;
                        SlopeVertexVsTrakX = SlopeVertexVsTrakX2;
                        SlopeVertexVsTrakY = SlopeVertexVsTrakY2;
                    }



                    double Sigma_slope = Sigma_calculation(tk.I_segments[0].sx - SlopeVertexVsTrakX, tk.I_segments[0].sy - SlopeVertexVsTrakY);

                    if (Sigma_slope <= slope_tolerance && flag == true)
                    {




                        Array.Resize(ref Indexes, Indexes.Length + 1);
                        Indexes[Indexes.Length - 1].id = tk.id;

                        Indexes[Indexes.Length - 1].id2 = tk.IpgId;

                        Indexes[Indexes.Length - 1].distance = Distance;
                        Indexes[Indexes.Length - 1].id_ref = index_1;
                        Indexes[Indexes.Length - 1].is_Vertex = false;
                        Indexes[Indexes.Length - 1].ref_is_vertex = false;
                        Indexes[Indexes.Length - 1].Sigma_Slope = Sigma_slope;
                        Indexes[Indexes.Length - 1].excl_tk_vx = true;
                    }



                }
            }
            if (Indexes.Length > 0)
            {
                int k;
                for (k = 0; k < Indexes.Length; k++)
                    if (Indexes[k].id != index_2)
                    {
                        if (Indexes[k].distance < Ip)
                            if (Indexes[k].Sigma_Slope < slope)
                            {
                                //Console.Write("C");
                                return false;

                            }
                    }
            }

            return true;
        }

        //---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------       
        //ricerca per vertici 
        static int Vtracks_search(Internal_I_Data tracks_volume_analysis, Int64 index, double sx, double sy, Versor Start_point, bool Vertex_Or_Track, double slope_tol, double sens_to_cooplanarity, int N_seg, double distance_cut, int N_plate, double exp_indexN)
        {
            Int32 i;

            Info_date[] Indexes = new Info_date[0];

            bool flag = true;
            bool operation = false;

            int recursion = -1;

            double exp_index = exp_indexN + 1;
            double vertex_X = Start_point.x;
            double vertex_Y = Start_point.y;
            double vertex_Z = Start_point.z;
            double slope_tolerance = 0;
            double SlopeVertexVsTrakX = 0;
            double SlopeVertexVsTrakY = 0;
            double SlopeVertexVsTrakX2 = 0;
            double SlopeVertexVsTrakY2 = 0;

            double Distance = 0;




            Versor Zero_TTrack;

            slope_tolerance = 0.2;



            for (i = 0; i < tracks_volume_analysis.I_tracks.Length; i++)
            {


                flag = false;

                Internal_Track tk = tracks_volume_analysis.I_tracks[i];


                if (Search_info_dates(tk.id, tk.IpgId, false) == false && tk.I_segments[tk.lenght - 1].z > Start_point.z && tk.lenght >= N_seg)
                {





                    Zero_TTrack = Fill_versor(tk.I_segments[tk.lenght - 1].x - Start_point.x, tk.I_segments[tk.lenght - 1].y - Start_point.y, tk.I_segments[tk.lenght - 1].z - Start_point.z);

                    Distance = Module_Calculation(Zero_TTrack);



                    if (Distance <= distance_cut)
                    {

                        SlopeVertexVsTrakX = Zero_TTrack.x / Zero_TTrack.z;
                        SlopeVertexVsTrakY = Zero_TTrack.y / Zero_TTrack.z;

                        if (Vertex_Or_Track == false)
                        {
                            SlopeVertexVsTrakX2 = sx;
                            SlopeVertexVsTrakY2 = sy;
                            double Dx;
                            Dx = Sigma_calculation(vertex_X - (tk.I_segments[tk.lenght - 1].sx * (vertex_Z - tk.I_segments[tk.lenght - 1].z) + tk.I_segments[tk.lenght - 1].x), vertex_Y - (tk.I_segments[tk.lenght - 1].sy * (vertex_Z - tk.I_segments[tk.lenght - 1].z) + tk.I_segments[tk.lenght - 1].y));
                            Distance = Dx;

                            if ((Sigma_calculation(SlopeVertexVsTrakX - SlopeVertexVsTrakX2, SlopeVertexVsTrakY - SlopeVertexVsTrakY2) <= slope_tolerance))
                            {
                                flag = true;
                            }
                        }
                        else flag = true;

                        double Sigma_slope = Sigma_calculation(tk.I_segments[tk.lenght - 1].sx - SlopeVertexVsTrakX, tk.I_segments[tk.lenght - 1].sy - SlopeVertexVsTrakY);

                        if (Sigma_slope <= 0.2 && flag == true)
                        //if(Tfake_search(tracks_volume_analysis, tk.id, index, 0.014, 40))
                        {


                            operation = true;



                            if (Vertex_Or_Track == false)
                            {
                                Array.Resize(ref Indexes, Indexes.Length + 1);
                                Indexes[Indexes.Length - 1].id = i;
                                Indexes[Indexes.Length - 1].id2 = Convert.ToInt64(exp_index);
                                Indexes[Indexes.Length - 1].distance = Distance;
                                Indexes[Indexes.Length - 1].id_ref = tk.id;
                                Indexes[Indexes.Length - 1].is_Vertex = false;
                                Indexes[Indexes.Length - 1].ref_is_vertex = false;
                                Indexes[Indexes.Length - 1].Sigma_Slope = Sigma_slope;
                                Indexes[Indexes.Length - 1].excl_tk_vx = true;
                            }
                        }

                    }

                }
            }
            if (Indexes.Length > 1)
            {
                int m;
                int recursion2 = 0;

                int k, j;
                long[] p = new long[2];
                p[0] = p[1] = -10;
                double Planar_value = sens_to_cooplanarity;
                double Planar_inter_value = 0;
                double Dx = 0;
                double Dx1 = 100000000000;
                double Dx2 = 0;


                for (k = 0; k < Indexes.Length; k++)
                {
                    for (j = 0; j < Indexes.Length - 1; j++)
                    {
                        if (j != k && Math.Abs(tracks_volume_analysis.I_tracks[Indexes[k].id].I_segments[tracks_volume_analysis.I_tracks[Indexes[k].id].lenght - 1].z - tracks_volume_analysis.I_tracks[Indexes[j].id].I_segments[tracks_volume_analysis.I_tracks[Indexes[j].id].lenght - 1].z) < 1000)
                        {

                            bool ys_1, ys_2, xs_1, xs_2;
                            double d_1, d_2, x1, x2, sx1, sx2, y1, y2, sy1, sy2, z1, z2;
                            x1 = tracks_volume_analysis.I_tracks[Indexes[k].id].I_segments[tracks_volume_analysis.I_tracks[Indexes[k].id].lenght - 1].x;
                            x2 = tracks_volume_analysis.I_tracks[Indexes[j].id].I_segments[tracks_volume_analysis.I_tracks[Indexes[j].id].lenght - 1].x;
                            y1 = tracks_volume_analysis.I_tracks[Indexes[k].id].I_segments[tracks_volume_analysis.I_tracks[Indexes[k].id].lenght - 1].y;
                            y2 = tracks_volume_analysis.I_tracks[Indexes[j].id].I_segments[tracks_volume_analysis.I_tracks[Indexes[j].id].lenght - 1].y;
                            z1 = tracks_volume_analysis.I_tracks[Indexes[k].id].I_segments[tracks_volume_analysis.I_tracks[Indexes[k].id].lenght - 1].z;
                            z2 = tracks_volume_analysis.I_tracks[Indexes[j].id].I_segments[tracks_volume_analysis.I_tracks[Indexes[j].id].lenght - 1].z;
                            sx1 = tracks_volume_analysis.I_tracks[Indexes[k].id].I_segments[tracks_volume_analysis.I_tracks[Indexes[k].id].lenght - 1].sx;
                            sx2 = tracks_volume_analysis.I_tracks[Indexes[j].id].I_segments[tracks_volume_analysis.I_tracks[Indexes[j].id].lenght - 1].sx;
                            sy1 = tracks_volume_analysis.I_tracks[Indexes[k].id].I_segments[tracks_volume_analysis.I_tracks[Indexes[k].id].lenght - 1].sy;
                            sy2 = tracks_volume_analysis.I_tracks[Indexes[j].id].I_segments[tracks_volume_analysis.I_tracks[Indexes[j].id].lenght - 1].sy;

                            d_1 = Sigma_calculation(x1 - x2, y1 - y2);
                            d_2 = Sigma_calculation(x1 - x2 + (sx1 - sx2) * 1300, y1 - y2 + (sy1 - sy2) * 1300);
                            if ((x1 - x2) >= 0) xs_1 = true; else xs_1 = false;
                            if ((y1 - y2) >= 0) ys_1 = true; else ys_1 = false;
                            if ((x1 - x2 + (sx1 - sx2) * 1300) >= 0) xs_2 = true; else xs_2 = false;
                            if ((y1 - y2 + (sy1 - sy2) * 1300) >= 0) ys_2 = true; else ys_2 = false;

                            if (d_1 < d_2 && xs_1 == xs_2 && ys_1 == ys_2 && d_1 < 65 && d_1 >= 1)//65
                            {

                                Internal_Segment Mid_Seg = new Internal_Segment();
                                Mid_Seg.x = (x1 + x2) / 2;
                                Mid_Seg.y = (y1 + y2) / 2;
                                Mid_Seg.z = (z1 + z2) / 2;
                                Versor Zero_VVertex;

                                Zero_VVertex = Versor_calculation(Fill_versor((Mid_Seg.x - Start_point.x), (Mid_Seg.y - Start_point.y), (Mid_Seg.z - Start_point.z)));

                                Mid_Seg.sx = Zero_VVertex.x / Zero_VVertex.z;
                                Mid_Seg.sy = Zero_VVertex.y / Zero_VVertex.z;

                                Planar_inter_value = Planarity_Check(Mid_Seg, tracks_volume_analysis.I_tracks[Indexes[k].id].I_segments[tracks_volume_analysis.I_tracks[Indexes[k].id].lenght - 1], tracks_volume_analysis.I_tracks[Indexes[j].id].I_segments[tracks_volume_analysis.I_tracks[Indexes[j].id].lenght - 1]);


                                Dx = Sigma_calculation(vertex_X - (((sx1 + sx2) / 2) * (vertex_Z - Mid_Seg.z) + Mid_Seg.x), vertex_Y - (((sy1 + sy2) / 2) * (vertex_Z - Mid_Seg.z) + Mid_Seg.y));




                                if (Planar_value > Planar_inter_value && Dx1 > Dx)
                                {
                                    Planar_value = Planar_inter_value;
                                    Dx1 = Dx;
                                    p[0] = j; p[1] = k;
                                    Dx2 = d_1;

                                }

                            }
                        }
                    }
                }

                for (k = 0; k < 2; k++)
                    if (p[k] != -10 && Search_info_dates(Indexes[p[k]].id_ref, Indexes[p[k]].id_ref, false) == false)
                    {
                        Fill_info_dates(Indexes[p[k]].id_ref, -Convert.ToInt64(exp_index), index, Indexes[p[k]].distance, false, Vertex_Or_Track, -Indexes[p[k]].Sigma_Slope, true);
                        Console.Write("+");

                        for (m = tracks_volume_analysis.I_tracks[Indexes[p[k]].id].lenght - 1; m >= 0; m--)
                        {


                            Versor Out_coord = new Versor();
                            Out_coord.x = tracks_volume_analysis.I_tracks[Indexes[p[k]].id].I_segments[m].x;
                            Out_coord.y = tracks_volume_analysis.I_tracks[Indexes[p[k]].id].I_segments[m].y;
                            Out_coord.z = tracks_volume_analysis.I_tracks[Indexes[p[k]].id].I_segments[m].z;

                            Indexes[p[k]].Sigma_Slope = Math.Abs(Planar_value);

                            //Indexes[p[k]].distance = Dx2;

                            recursion = tracks_search(tracks_volume_analysis, Indexes[p[k]].id_ref, tracks_volume_analysis.I_tracks[Indexes[p[k]].id].I_segments[m].sx, tracks_volume_analysis.I_tracks[Indexes[p[k]].id].I_segments[m].sy, Out_coord, false, slope_tol, sens_to_cooplanarity, 1, N_plate * 1500, N_plate, exp_index, 0, 0, 0);
                            if (recursion > recursion2)
                                recursion2 = recursion;

                        }
                    }
            }

            if (operation == true)
                return 1;
            else
                return 0;
        }

        //---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        //ricerca per tracce 
        static int tracks_search(Internal_I_Data tracks_volume_analysis, Int64 index, double sx, double sy, Versor Start_point, bool Vertex_Or_Track, double slope_tol, double sens_to_cooplanarity, int N_seg, double distance_cut, int N_plate, double exp_indexN, double Zmin, double s_sx, double s_sy)
        {
            Int32 i;

            Info_date[] Indexes = new Info_date[0];

            bool flag = true;
            bool operation = false;

            int recursion = -1;

            double exp_index = exp_indexN + 1;
            double vertex_X = Start_point.x;
            double vertex_Y = Start_point.y;
            double vertex_Z = Start_point.z;
            double slope_tolerance = 0;
            double SlopeVertexVsTrakX = 0;
            double SlopeVertexVsTrakY = 0;
            double SlopeVertexVsTrakX2 = 0;
            double SlopeVertexVsTrakY2 = 0;

            double Distance = 0;




            Versor Zero_TTrack;



            if (!Vertex_Or_Track)
            {


                slope_tolerance = slope_tol;
            }
            else slope_tolerance = 0.12;




            for (i = 0; i < tracks_volume_analysis.I_tracks.Length; i++)
            {

                flag = false;

                Internal_Track tk = tracks_volume_analysis.I_tracks[i];

                if (Search_info_dates(tk.id, tk.IpgId, false) == false && tk.lenght >= N_seg && tk.I_segments[tk.lenght - 1].z > Start_point.z)
                {

                    Zero_TTrack = Fill_versor(tk.I_segments[tk.lenght - 1].x - Start_point.x, tk.I_segments[tk.lenght - 1].y - Start_point.y, tk.I_segments[tk.lenght - 1].z - Start_point.z);

                    Distance = Module_Calculation(Zero_TTrack);

                    if (Distance <= distance_cut)
                    {

                        SlopeVertexVsTrakX = Zero_TTrack.x / Zero_TTrack.z;
                        SlopeVertexVsTrakY = Zero_TTrack.y / Zero_TTrack.z;

                        double Dx;
                        Dx = Sigma_calculation(vertex_X - (tk.I_segments[tk.lenght - 1].sx * (vertex_Z - tk.I_segments[tk.lenght - 1].z) + tk.I_segments[tk.lenght - 1].x), vertex_Y - (tk.I_segments[tk.lenght - 1].sy * (vertex_Z - tk.I_segments[tk.lenght - 1].z) + tk.I_segments[tk.lenght - 1].y));
                        Distance = Dx;

                        if (Vertex_Or_Track == false)
                        {
                            SlopeVertexVsTrakX2 = sx;
                            SlopeVertexVsTrakY2 = sy;

                            if ((Sigma_calculation(SlopeVertexVsTrakX - SlopeVertexVsTrakX2, SlopeVertexVsTrakY - SlopeVertexVsTrakY2) <= slope_tolerance))
                            {
                                flag = true;
                                SlopeVertexVsTrakX = SlopeVertexVsTrakX2;
                                SlopeVertexVsTrakY = SlopeVertexVsTrakY2;
                            }

                        }
                        else flag = true;

                        double Sigma_slope = Sigma_calculation(tk.I_segments[tk.lenght - 1].sx - SlopeVertexVsTrakX, tk.I_segments[tk.lenght - 1].sy - SlopeVertexVsTrakY);

                        if (Sigma_slope <= slope_tolerance && flag == true)
                        {


                            if (Vertex_Or_Track == true && tk.I_segments[tk.lenght - 1].z > Zmin)
                            {
                                if (Distance < 2100)
                                {

                                    double X_cone = 50;
                                    double Intercept_X, Intercept_Y;
                                    Intercept_X = vertex_X - s_sx * vertex_Z;
                                    Intercept_Y = vertex_Y - s_sy * vertex_Z;

                                    if (Math.Abs(tk.I_segments[tk.lenght - 1].x - s_sx * tk.I_segments[tk.lenght - 1].z - Intercept_X) < X_cone && Math.Abs(tk.I_segments[tk.lenght - 1].y - s_sy * tk.I_segments[tk.lenght - 1].z - Intercept_Y) < X_cone)
                                    {
                                        int m;
                                        int recursion2 = 0;

                                        if (Search_info_dates(tk.id, tk.IpgId, false) == false)// && (recursion2 == 1 || tk.lenght > 1))
                                        {

                                            Fill_info_dates(tk.id, Convert.ToInt64(exp_index), index, Distance, false, Vertex_Or_Track, Sigma_slope, true);
                                            operation = true;
                                            Console.Write("X");



                                            for (m = tk.lenght - 1; m >= 0; m--)
                                            {
                                                Versor Out_coord = new Versor();
                                                Out_coord.x = tk.I_segments[m].x;
                                                Out_coord.y = tk.I_segments[m].y;
                                                Out_coord.z = tk.I_segments[m].z;
                                                recursion = tracks_search(tracks_volume_analysis, tk.id, tk.I_segments[m].sx, tk.I_segments[m].sy, Out_coord, false, slope_tol, sens_to_cooplanarity, 1, N_plate * 1500, N_plate, exp_index, 0, 0, 0);
                                                Vtracks_search(tracks_volume_analysis, tk.id, tk.I_segments[m].sx, tk.I_segments[m].sy, Out_coord, false, slope_tol, sens_to_cooplanarity, 1, 20 * 1500, N_plate, exp_index);


                                                if (recursion > recursion2)
                                                    recursion2 = recursion;
                                            }
                                        }
                                        //if (Search_info_dates(tk.id, tk.IpgId, false) == false && (recursion2 == 1 || tk.lenght > 1))
                                        //{
                                        //    if (tk.IpgId > 0)
                                        //        Fill_info_dates(tk.id, Convert.ToInt64(exp_index), index, Distance, false, Vertex_Or_Track, Sigma_slope, true);
                                        //    else
                                        //        Fill_info_dates(tk.id, -Convert.ToInt64(exp_index), index, Distance, false, Vertex_Or_Track, Sigma_slope, true);
                                        //    operation = true;
                                        //    Console.Write("-");

                                        //}
                                        //if (recursion2 == 1 || tk.lenght > 1)
                                        //{
                                        //Fill_info_dates(tk.id, tk.IpgId, index, Distance, false, Vertex_Or_Track, Sigma_slope, true);

                                    }
                                }
                            }


                            if (Vertex_Or_Track == false)
                            {
                                if (Distance < 200)
                                {

                                    double Sigma_slope_send, Distance_send;
                                    Sigma_slope_send = 0.02;
                                    Distance_send = 40;
                                    if (Sigma_slope < 0.02)
                                        Sigma_slope_send = Sigma_slope;
                                    if (Distance < 40)
                                        Distance_send = Distance;

                                    Versor Out_coord = new Versor();

                                    Out_coord.x = tk.I_segments[tk.lenght - 1].x;
                                    Out_coord.y = tk.I_segments[tk.lenght - 1].y;
                                    Out_coord.z = tk.I_segments[tk.lenght - 1].z;

                                    if (Tfake_search(tracks_volume_analysis, tk.id, index, Sigma_slope_send, Distance_send, tk.I_segments[tk.lenght - 1].sx, tk.I_segments[tk.lenght - 1].sy, Out_coord))// || !pair_track_search(index))
                                    {

                                        int m;
                                        int recursion2 = 0;


                                        if (Search_info_dates(tk.id, tk.IpgId, false) == false)
                                            if (Sigma_slope > Distance * ((0.17 + 0.07) / 220) - 0.07)
                                            {


                                                Fill_info_dates(tk.id, Convert.ToInt64(exp_index), index, Distance, false, Vertex_Or_Track, Sigma_slope, true);

                                                operation = true;
                                                Console.Write("-");


                                                for (m = tk.lenght - 1; m >= 0; m--)
                                                {

                                                    Out_coord.x = tk.I_segments[m].x;
                                                    Out_coord.y = tk.I_segments[m].y;
                                                    Out_coord.z = tk.I_segments[m].z;

                                                    if (exp_index < 11)
                                                    {

                                                        Vtracks_search(tracks_volume_analysis, tk.id, tk.I_segments[m].sx, tk.I_segments[m].sy, Out_coord, false, slope_tol, sens_to_cooplanarity, 1, 20 * 1500, N_plate, exp_index);
                                                    }
                                                    recursion = tracks_search(tracks_volume_analysis, tk.id, tk.I_segments[m].sx, tk.I_segments[m].sy, Out_coord, false, slope_tol, sens_to_cooplanarity, 1, N_plate * 1500, N_plate, exp_index, 0, 0, 0);
                                                    if (recursion > recursion2)
                                                        recursion2 = recursion;
                                                }
                                            }
                                    }

                                }

                            }

                        }

                    }

                }

            }

            if (operation == true)
                return 1;
            else
                return 0;

        }

        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------
        //organizzazione dati interni da ascii--------------------------------------------------------------------------------------------------------------------------------
        static Internal_I_Data Ascii_s(string name_file, double z)
        {
            Internal_I_Data I_data = new Internal_I_Data();

            I_data.I_tracks = new Internal_Track[0];
            using (StreamReader sr = new StreamReader(name_file))
            {
                String line;
                int FirstTab, LastTab;
                int i = 0;
                while (!sr.EndOfStream)
                {
                    line = sr.ReadLine();
                    if (line.Length == 0)
                        break;


                    FirstTab = LastTab = 0;
                    int k = 0;

                    double PdgId = 0;
                    double X = 0;
                    double Y = 0;
                    double Z = 0;
                    double sx = 0;
                    double sy = 0;
                    bool flag = false;
                    while (k < 7)
                    {
                        while (line[FirstTab] == ' ' || line[FirstTab] == '\t')
                        {
                            FirstTab++;
                            if (FirstTab >= line.Length) break;
                        }


                        LastTab = FirstTab + 1;
                        if (LastTab == line.Length)
                            LastTab = LastTab - 1;

                        if (FirstTab >= line.Length) break;
                        while (line[LastTab] != ' ' && line[LastTab] != '\t')
                        {
                            LastTab++;
                            if (LastTab >= line.Length) break;
                        }
                        if (LastTab >= line.Length)
                            LastTab = LastTab - 1;

                        if (k == 1)
                            PdgId = System.Convert.ToDouble(line.Substring(FirstTab, LastTab - FirstTab));
                        if (k == 2)
                        {
                            X = System.Convert.ToDouble(line.Substring(FirstTab, LastTab - FirstTab));
                        }
                        if (k == 3)
                        {
                            Y = System.Convert.ToDouble(line.Substring(FirstTab, LastTab - FirstTab));
                        }
                        if (k == 4)
                        {
                            sx = System.Convert.ToDouble(line.Substring(FirstTab, LastTab - FirstTab));
                        }
                        if (k == 5)
                        {
                            sy = System.Convert.ToDouble(line.Substring(FirstTab, LastTab - FirstTab));
                        }
                        if (k == 6)
                        {
                            //Z = 1300*(System.Convert.ToDouble(line.Substring(FirstTab, LastTab - FirstTab+1))-1);
                            Z = System.Convert.ToDouble(line.Substring(FirstTab, LastTab - FirstTab + 1));
                            if (Z > z)
                                flag = true;
                        }
                        FirstTab = LastTab + 1;
                        k++;
                    }
                    //if (flag == true && PdgId>=0)
                    if (flag == true)
                    {
                        Array.Resize(ref I_data.I_tracks, I_data.I_tracks.Length + 1);
                        I_data.I_tracks[i].id = i;
                        I_data.I_tracks[i].IpgId = Convert.ToInt16(PdgId);
                        I_data.I_tracks[i].is_BaseTrack = false;
                        I_data.I_tracks[i].lenght = 1;
                        I_data.I_tracks[i].I_segments = new Internal_Segment[1];
                        I_data.I_tracks[i].I_segments[0].id = 0;
                        I_data.I_tracks[i].I_segments[0].x = X;
                        I_data.I_tracks[i].I_segments[0].y = Y;
                        I_data.I_tracks[i].I_segments[0].z = Z;
                        I_data.I_tracks[i].I_segments[0].sx = sx;
                        I_data.I_tracks[i].I_segments[0].sy = sy;

                        i++;
                    }
                }
                sr.Close();
            }

            return I_data;
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------
        //organizzazione dati interni da tsr volume track---------------------------------------------------------------------------------------------------------------------
        static Internal_I_Data TSR_I(SySal.TotalScan.Volume v, Versor XYZ, double Zmin, double s_sx, double s_sy)
        {
            int i, k, l;
            int Tr_len;
            Internal_I_Data I_data = new Internal_I_Data();


            Tr_len = v.Tracks.Length;



            l = 0;
            I_data.I_tracks = new Internal_Track[0];
            for (i = 0; i < Tr_len; i++)
            {

                double X, Y, Z;
                double Intercept_X, Intercept_Y;

                X = v.Tracks[i][v.Tracks[i].Length - 1].Info.Intercept.X;
                Y = v.Tracks[i][v.Tracks[i].Length - 1].Info.Intercept.Y;
                Z = v.Tracks[i][v.Tracks[i].Length - 1].Info.Intercept.Z;

                Intercept_X = XYZ.x - s_sx * XYZ.z;
                Intercept_Y = XYZ.y - s_sy * XYZ.z;

                if (Z >= Zmin && Math.Abs(X - s_sx * Z - Intercept_X) < 2500 && Math.Abs(Y - s_sy * Z - Intercept_Y) < 2500)
                {

                    Array.Resize(ref I_data.I_tracks, I_data.I_tracks.Length + 1);
                    I_data.I_tracks[l].I_segments = new Internal_Segment[v.Tracks[i].Length];
                    I_data.I_tracks[l].id = v.Tracks[i].Id;
                    I_data.I_tracks[l].lenght = v.Tracks[i].Length;
                    I_data.I_tracks[l].is_BaseTrack = false;
                    I_data.I_tracks[l].IpgId = l;
                    for (k = 0; k < v.Tracks[i].Length; k++)
                    {
                        I_data.I_tracks[l].I_segments[k].id = k;
                        I_data.I_tracks[l].I_segments[k].x = v.Tracks[i][k].Info.Intercept.X;
                        I_data.I_tracks[l].I_segments[k].y = v.Tracks[i][k].Info.Intercept.Y;
                        I_data.I_tracks[l].I_segments[k].z = v.Tracks[i][k].Info.Intercept.Z;
                        I_data.I_tracks[l].I_segments[k].sx = v.Tracks[i][k].Info.Slope.X;
                        I_data.I_tracks[l].I_segments[k].sy = v.Tracks[i][k].Info.Slope.Y;
                    }
                    l++;
                }

            }
            return I_data;
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------       
        //registra un file ascii da dati simulati-----------------------------------------------------------------------------------------------------------------------------
        static void IO_data_Ascii_s(Internal_I_Data V, string Out_name_file)
        {
            int i;

            using (StreamWriter sw = new StreamWriter(Out_name_file))
            {

                sw.WriteLine("Id\told_Id\tRindex\tX\tY\tZ\tSX\tSY\tIp\tsigma");

                for (i = 0; i < info_t_v.Length; i++)
                {


                    sw.WriteLine(info_t_v[i].id + "\t" + info_t_v[i].id_ref + "\t" + info_t_v[i].id2 + "\t" + V.I_tracks[info_t_v[i].id].I_segments[0].x + "\t" + V.I_tracks[info_t_v[i].id].I_segments[0].y +
                            "\t" + (V.I_tracks[info_t_v[i].id].I_segments[0].z) + "\t" + V.I_tracks[info_t_v[i].id].I_segments[0].sx +
                            "\t" + V.I_tracks[info_t_v[i].id].I_segments[0].sy + "\t" + info_t_v[i].distance + "\t" + info_t_v[i].Sigma_Slope);

                }
                Console.WriteLine("");
                Console.WriteLine(info_t_v.Length + " tracks found");
                Console.WriteLine("file wrote: " + Out_name_file);

                sw.Flush();
                sw.Close();
            }


        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------
        //registra un file tsr------------------------------------------------------------------------------------------------------------------------------------------------
        static void IO_data_TSR(SySal.TotalScan.Volume vol, string Out_name_file)
        {
            int i, j;

            MyVolume V_new2 = new MyVolume(-1, true, vol, vol);

            //vol.Layers[0][9].Info.Intercept.X
            using (StreamWriter sc = new StreamWriter(Out_name_file + ".txt"))
            {

                for (i = 0; i < info_t_v.Length; i++)
                {



                    V_new2 = new MyVolume(Convert.ToInt32(info_t_v[i].id), info_t_v[i].is_Vertex, V_new2, vol);


                    if (info_t_v[i].is_Vertex == false)
                    {
                        V_new2.Tracks[V_new2.Tracks.Length - 1].SetAttribute(new SySal.TotalScan.NamedAttributeIndex("sigma"), info_t_v[i].Sigma_Slope);


                        V_new2.Tracks[V_new2.Tracks.Length - 1].SetAttribute(new SySal.TotalScan.NamedAttributeIndex("distance from track"), info_t_v[i].distance);
                        V_new2.Tracks[V_new2.Tracks.Length - 1].SetAttribute(new SySal.TotalScan.NamedAttributeIndex("old id track ref"), info_t_v[i].id_ref);

                        //sc.WriteLine(1 + "\t" + info_t_v[i].id2 + "\t" + info_t_v[i].Sigma_Slope + "\t" + info_t_v[i].distance + "\t" + V_new2.Tracks[V_new2.Tracks.Length - 1].Length + "\t" + info_t_v[i].id + "\t" + info_t_v[i].id_ref);
                    }

                    //}
                }
                sc.WriteLine("id\tx\ty\tz\tsx\tsy");
                for (i = 0; i < V_new2.Tracks.Length; i++)
                    for (j = 0; j < V_new2.Tracks[i].Length; j++)
                        sc.WriteLine(V_new2.Tracks[i].Id + "\t" + V_new2.Tracks[i][j].Info.Intercept.X + "\t" + V_new2.Tracks[i][j].Info.Intercept.Y + "\t" + V_new2.Tracks[i][j].Info.Intercept.Z + "\t" + V_new2.Tracks[i][j].Info.Slope.X + "\t" + V_new2.Tracks[i][j].Info.Slope.Y);

                sc.Flush();
                sc.Close();
            }

            SySal.OperaPersistence.Persist((Out_name_file + ".tsr"), V_new2);

            //scrive su console le informazioni relative all'elaborazione
            Console.WriteLine();
            Console.WriteLine("Vertices found : " + V_new2.Vertices.Length);
            Console.WriteLine("Track found : " + V_new2.Tracks.Length);
            Console.WriteLine("File written: " + Out_name_file + ".tsr");
            Console.WriteLine("File written: " + Out_name_file + ".txt");
            Console.Beep(2000, 50);
            Console.Beep(3000, 50);

        }

        //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        //Struttura dati interna------------------------------------------------------------------------------------------------------------------------------------------------------

        public struct Internal_I_Data
        {

            public Internal_Track[] I_tracks;
        }

        public struct Internal_Track
        {
            public Internal_Segment[] I_segments;
            public Int64 id;
            public int lenght;
            public Int64 IpgId;
            public bool is_BaseTrack;
        }
        public struct Internal_Segment
        {
            public double x, y, z, sx, sy;
            public Int64 id;
        }
        //---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        static bool pair_track_search(Int64 id)
        {
            int i;
            for (i = 0; i < info_t_v.Length; i++)
            {
                if (info_t_v[i].id == id)
                    if (info_t_v[i].Sigma_Slope < 0)
                        return false;
            }
            return true;
        }
        //----------------------------------------------------------------------------------------------------------------------------------------------------
        static void Fill_info_dates(Int64 id, Int64 id2, Int64 id_ref, double distance, bool is_vertex, bool ref_is_vertex, double Sigma_Slope, bool tk_vx)
        {
            Array.Resize(ref info_t_v, info_t_v.Length + 1);
            info_t_v[info_t_v.Length - 1].id = id;
            info_t_v[info_t_v.Length - 1].id2 = id2;
            info_t_v[info_t_v.Length - 1].id_ref = id_ref;
            info_t_v[info_t_v.Length - 1].is_Vertex = is_vertex;
            info_t_v[info_t_v.Length - 1].distance = distance;
            info_t_v[info_t_v.Length - 1].ref_is_vertex = ref_is_vertex;
            info_t_v[info_t_v.Length - 1].Sigma_Slope = Sigma_Slope;
            info_t_v[info_t_v.Length - 1].excl_tk_vx = tk_vx;
        }
        //----------------------------------------------------------------------------------------------------------------------------------------------------
        static bool Search_info_dates(Int64 id, Int64 id2, bool is_vertex)
        {
            int i;
            for (i = 0; i < info_t_v.Length; i++)
            {
                if (info_t_v[i].id == id && info_t_v[i].is_Vertex == is_vertex)
                    return true;
            }
            return false;
        }
        //----------------------------------------------------------------------------------------------------------------------------------------------------
        public struct Info_date
        {
            public Int64 id;
            public Int64 id2;
            public bool is_Vertex;
            public bool ref_is_vertex;
            public Int64 id_ref;
            public double distance;
            public double Sigma_Slope;
            public bool excl_tk_vx;

        }
        //----------------------------------------------------------------------------------------------------------------------------------------------------

        // Funzioni di calcolo vettoriale
        //------------------------------------------------------------------------------------------
        static double Sigma_calculation(double sx, double sy)
        {
            double s = Math.Sqrt(sx * sx + sy * sy);
            return s;

        }
        //------------------------------------------------------------------------------------------
        // calcola prodotto vettoriale
        static Versor VectorialProduct(Versor a, Versor b)
        {
            Versor c;
            double module;

            c.x = a.y * b.z - a.z * b.y;
            c.y = a.z * b.x - a.x * b.z;
            c.z = a.x * b.y - a.y * b.x;

            module = Module_Calculation(c);

            c = Versor_calculation(c);

            return c;
        }
        //------------------------------------------------------------------------------------------
        //calcola prodotto scalare
        static double ScalarProduct(Versor a, Versor b)
        {

            double scalar_product_value = a.x * b.x + a.y * b.y + a.z * b.z;

            return scalar_product_value;
        }
        //------------------------------------------------------------------------------------------
        //calcola versore
        static Versor Versor_calculation(Versor a)
        {
            Versor b;

            double module = Module_Calculation(a);

            b.x = a.x / module;
            b.y = a.y / module;
            b.z = a.z / module;

            return b;
        }
        //------------------------------------------------------------------------------------------
        //calcola la planarità tra tre vettori
        static double Planarity_Check(Internal_Segment a, Internal_Segment b, Internal_Segment c)
        {

            Versor[] Tracks_unity_vector = new Versor[4];

            Tracks_unity_vector[0] = Versor_calculation(Fill_versor(a.sx, a.sy, 1));
            Tracks_unity_vector[1] = Versor_calculation(Fill_versor(b.sx, b.sy, 1));
            Tracks_unity_vector[2] = Versor_calculation(Fill_versor(c.sx, c.sy, 1));

            Tracks_unity_vector[3] = VectorialProduct(Tracks_unity_vector[1], Tracks_unity_vector[2]);
            return Math.Abs(ScalarProduct(Tracks_unity_vector[0], Tracks_unity_vector[3]));
        }
        //------------------------------------------------------------------------------------------
        // calcola il modulo del vettore
        static double Module_Calculation(Versor a)
        {

            double module = Math.Sqrt(a.x * a.x + a.y * a.y + a.z * a.z);

            return module;
        }
        //------------------------------------------------------------------------------------------
        //riempie un vettore
        static Versor Fill_versor(double ax, double ay, double az)
        {
            Versor b;
            b.x = ax;
            b.y = ay;
            b.z = az;

            return b;
        }

        public struct Versor
        {
            public double x;
            public double y;
            public double z;
        }

    }
    //-------------------------------------------------------------------------------------------------------------------------------
    //TSR INTERNAL CLASS
    //-------------------------------------------------------------------------------------------------------------------------------

    internal class MyLayerList : SySal.TotalScan.Volume.LayerList
    {
        public MyLayerList(SySal.TotalScan.Volume old_volume2)
        {
            int t;
            this.Items = new SySal.TotalScan.Layer[old_volume2.Layers.Length];
            for (t = 0; t < old_volume2.Layers.Length; t++)
                Items[t] = old_volume2.Layers[t];
        }
    }
    internal class MySegments : SySal.TotalScan.Segment
    {
        public MySegments(SySal.TotalScan.Segment t)
        {
            this.m_Index = t.Index;
            this.m_Info = t.Info;
            this.m_LayerOwner = t.LayerOwner;
            this.m_PosInLayer = t.PosInLayer;
            this.m_PosInTrack = t.PosInTrack;
            this.m_TrackOwner = t.TrackOwner;
        }
    }
    internal class MyTrakListId : SySal.TotalScan.Track
    {
        public MyTrakListId(int i, SySal.TotalScan.Track t)
        {
            int k, l;
            l = t.Length - 1;
            for (k = l; k >= 0; k--)
            {
                //SySal.TotalScan.Track TK = t;
                //TK = t[k];
                this.AddSegment(t[k]);
            }
            this.m_AttributeList = t.ListAttributes();
            this.m_Downstream_Layer = t.DownstreamLayer;
            this.m_Downstream_LayerId = t.DownstreamLayerId;
            this.m_Downstream_Vertex = t.Downstream_Vertex;
            this.m_Id = i;
            this.m_Upstream_Layer = t.UpstreamLayer;
            this.m_Upstream_LayerId = t.UpstreamLayerId;
            this.m_Upstream_Vertex = t.Upstream_Vertex;

            if (t.Length > 1)
            {
                this.m_Downstream_PosX = t.Downstream_PosX;
                this.m_Downstream_PosY = t.Downstream_PosY;
                this.m_Downstream_SlopeX = t.Downstream_SlopeX;
                this.m_Downstream_SlopeY = t.Downstream_SlopeY;
                this.m_Upstream_PosX = t.Upstream_PosX;
                this.m_Upstream_PosY = t.Upstream_PosY;
                this.m_Upstream_SlopeX = t.Upstream_SlopeX;
                this.m_Upstream_SlopeY = t.Upstream_SlopeY;

                if (t.Downstream_Vertex != null)
                    this.m_Downstream_Impact_Parameter = t.Downstream_Impact_Parameter;
                else
                    this.m_Downstream_Impact_Parameter = -1;

                if (t.Upstream_Vertex != null)
                    this.m_Upstream_Impact_Parameter = t.Upstream_Impact_Parameter;
                else
                    this.m_Upstream_Impact_Parameter = -1;
            }

            SetAttribute(new SySal.TotalScan.NamedAttributeIndex("old_id"), t.Id);

            //int k, l;
            //l = t.Length;

            //for (k = 0; k < l; k++)
            //{
            //    //SySal.TotalScan.Track TK = t;
            //    //TK = t[k];
            //    this[k]=new MySegments(t[k]);
            //}
            //this.AddSegmentAndCheck(t[k]);
        }
    }

    internal class MyTrackList : SySal.TotalScan.Volume.TrackList
    {
        public MyTrackList(int i, bool is_vertex, SySal.TotalScan.Volume vlist, SySal.TotalScan.Volume vlist2)
        {
            int t = 0;
            if (is_vertex == true)
            {
                if (i > -1)
                {

                    this.Items = new SySal.TotalScan.Track[vlist.Tracks.Length + vlist2.Vertices[i].Length];




                    if (vlist.Tracks.Length > 0)
                    {
                        for (t = 0; t < vlist.Tracks.Length; t++)
                        {
                            this.Items[t] = vlist.Tracks[t];

                        }
                        for (t = 0; t < vlist2.Vertices[i].Length; t++)
                            this.Items[vlist.Tracks.Length + t] = new MyTrakListId(vlist.Tracks.Length + t, vlist2.Vertices[i][t]);
                    }
                    else
                    {
                        for (t = 0; t < vlist2.Vertices[i].Length; t++)
                            this.Items[vlist.Tracks.Length + t] = new MyTrakListId(vlist.Tracks.Length + t, vlist2.Vertices[i][t]);
                    }
                }



                if (i == -2)
                {
                    this.Items = new SySal.TotalScan.Track[vlist2.Tracks.Length];
                    int k;
                    for (k = 0; k < vlist2.Tracks.Length; k++)
                        this.Items[k] = vlist2.Tracks[k];
                }
                if (i == -1)
                    this.Items = new SySal.TotalScan.Track[0];
            }
            else
            {
                this.Items = new SySal.TotalScan.Track[vlist.Tracks.Length + 1];
                for (t = 0; t < vlist.Tracks.Length; t++)
                {
                    this.Items[t] = vlist.Tracks[t];

                }
                this.Items[vlist.Tracks.Length] = new MyTrakListId(vlist.Tracks.Length, vlist2.Tracks[i]);
                //int j;
                //for (j = 0; j < vlist2.Tracks[i].Length; j++)
                //    this.Items[vlist.Tracks.Length].AddSegment(vlist2.Tracks[i][j]);
            }

        }

    }

    internal class MyVertexListId : SySal.TotalScan.Vertex
    {
        public MyVertexListId(int i, SySal.TotalScan.Vertex v, SySal.TotalScan.Volume vlist_updated)
        {
            int k;

            for (k = (v.Length - 1); k >= 0; k--)
                this.AddTrack(vlist_updated.Tracks[vlist_updated.Tracks.Length - v.Length + k], false);


            this.m_AverageDistance = v.AverageDistance;
            this.m_DX = v.DX;
            this.m_DY = v.DY;
            this.m_X = v.X;
            this.m_Y = v.Y;
            this.m_Z = v.Z;
            this.m_Id = i;
            SetAttribute(new SySal.TotalScan.NamedAttributeIndex("old_id"), v.Id);
        }
    }


    internal class MyVertexList : SySal.TotalScan.Volume.VertexList
    {
        public MyVertexList(int i, SySal.TotalScan.Volume vlist_updated, SySal.TotalScan.Volume vlist, SySal.TotalScan.Volume vlist2)
        {
            int t = 0;
            if (i > -1)
            {
                this.Items = new SySal.TotalScan.Vertex[vlist.Vertices.Length + 1];


                if (vlist.Vertices.Length > 0)
                {
                    for (t = 0; t < vlist.Vertices.Length; t++)
                        this.Items[t] = vlist.Vertices[t];

                    this.Items[vlist.Vertices.Length] = new MyVertexListId(vlist.Vertices.Length, vlist2.Vertices[i], vlist_updated);
                }
                else { this.Items[0] = new MyVertexListId(0, vlist2.Vertices[i], vlist_updated); }
            }
            else if (i == -1)
                this.Items = new SySal.TotalScan.Vertex[0];
            else if (i == -2)
            {
                this.Items = new SySal.TotalScan.Vertex[vlist.Vertices.Length];

                for (t = 0; t < vlist.Vertices.Length; t++)
                    this.Items[t] = vlist.Vertices[t];
            }
        }
    }

    internal class MyVolume : SySal.TotalScan.Volume
    {
        public MyVolume(int idvtxtoexclude, bool IsVertex, SySal.TotalScan.Volume oldvol, SySal.TotalScan.Volume oldvol2)
        {



            if (idvtxtoexclude == -1)
            {
                this.m_Layers = new MyLayerList(oldvol2);
                this.m_Vertices = new MyVertexList(-1, oldvol, oldvol, oldvol2);
                this.m_Tracks = new MyTrackList(-1, true, oldvol, oldvol2);
            }
            else if (idvtxtoexclude > -1)
            {

                if (IsVertex == true)
                {
                    this.m_Tracks = new MyTrackList(oldvol2.Vertices[idvtxtoexclude].Id, true, oldvol, oldvol2);
                    this.m_Vertices = new MyVertexList(idvtxtoexclude, this, oldvol, oldvol2);
                    this.m_Layers = new MyLayerList(oldvol2);
                }
                else
                {
                    this.m_Tracks = new MyTrackList(idvtxtoexclude, false, oldvol, oldvol2);
                    this.m_Layers = new MyLayerList(oldvol2);
                    this.m_Vertices = new MyVertexList(-2, oldvol, oldvol, oldvol);
                }
            }
        }
    }
}
