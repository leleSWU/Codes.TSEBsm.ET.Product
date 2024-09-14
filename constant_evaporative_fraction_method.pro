pro calc_LEarea2_inputSdn_PT_xiugai,cosza
  year = ''
  dirin = file_search("")         
  dirin2 = file_search("")        
  dirin3 = file_search('")       
  
  model_input = ""
  output_d = ''
  
  dirin_HC = file_search(model_input + '\Hc_2004\', "*.tiff")   
  dirin_Hs = file_search(model_input + '\Hs_2004\', "*.tiff")  
  dirin_LEC = file_search(model_input + '\LEc_2004\', "*.tiff") 
  dirin_LES = file_search(model_input + '\LEs_2004\', "*.tiff")   

  dirout1 = output_d + '\daily\EF_s\'
  dirout2 = output_d + '\daily\LE_S\'
  dirout3 = output_d + '\daily\LE_C\'
  dirout4= output_d + '\daily\EF_C\'

  dirin_Sdn = file_search( "")
  dirin_landcover = ""
  dirin_landcover = ""
  dirout_LE=output_d + '\daily\LE\'
  dirout_Rn=output_d + '\daily\Rn\'
  dirout_Rn_c=output_d + '\daily\Rn_C\'
  dirout_Rn_s=output_d + '\daily\Rn_S\'
  
  if file_test(dirout1) eq 0 then file_mkdir,dirout1
  if file_test(dirout2) eq 0 then file_mkdir,dirout2
  if file_test(dirout3) eq 0 then file_mkdir,dirout3
  if file_test(dirout4) eq 0 then file_mkdir,dirout4
  
  if file_test(dirout_LE) eq 0 then file_mkdir,dirout_LE
  if file_test(dirout_Rn) eq 0 then file_mkdir,dirout_Rn
  if file_test(dirout_Rn_c) eq 0 then file_mkdir,dirout_Rn_c
  if file_test(dirout_Rn_s) eq 0 then file_mkdir,dirout_Rn_s
  
  print,1
  file_year_doy = strsplit(file_basename(dirin),".",/extract)
  year_doys = strarr(n_elements(file_year_doy))
  for i=0,n_elements(file_year_doy)-1 do begin
    temp = file_year_doy[i]
    year_doys[i] = temp[0]
  endfor
  year_doys = strsplit(year_doys, "_", /extract)
  doys = strarr(n_elements(file_year_doy))
  for i=0,n_elements(file_year_doy)-1 do begin
    temp = year_doys[i]
    doys[i] = strmid((temp[2]),4,3)
  endfor

  radperdegree=0.01745
  albedo = 0.2
  time=14 
  landcover_data = read_tiff(dirin_landcover)

  day_num = n_elements(doys)
  for i = 0, day_num do begin
    ;LEc
    in_file = dirin_LEC[i]
    LE_c = read_tiff(in_file)      
    
    in_file = dirin_HC[i]
    H_c = read_tiff(in_file)     
    
    EF_c = LE_c/(LE_c+H_c)
    EF_c= (EF_c lt 0.01)*0.01+(EF_c ge 0.01)*EF_c
    ;LEs
    in_file = dirin_LES[i]
    LE_s = read_tiff(in_file)      
    in_file = dirin_Hs[i]
    H_s = read_tiff(in_file)      

    EF_s = LE_s/(LE_s+H_s)

    in_file = dirin[i]
    rlai = read_tiff(in_file)      
    rlai[where(rlai gt 100)] = 1.0
    fc = 1- exp(-0.5*rlai)
    fc[where(fc lt 0)] = 0
    
    EF_s[where(EF_s le 0.01)] = 0.3
    EF_s[where(rlai lt 0.2 and EF_s gt 0.3)] = 0.3
    EF_s[where(EF_s gt 1)] = 1.0

    in_file =dirin2[i]
    ta = read_tiff(in_file)
    ta = ta - 273.15
    Tak = ta + 273.15

    ta[where(ta lt 0)] = 0

    es = 6.112*exp((17.67*(ta))/(ta+243.5))


    in_file = dirin3[1]
    dem = read_tiff(in_file)

    in_file = dirin3[2]
    slope = read_tiff(in_file)

    in_file = dirin3[0]
    aspect = read_tiff(in_file)

    in_file_lat = "Q:\G\Data_Input_instantaneous_daily_1°\lat\lat_pre_73.0_55.0_135.0_16.0.tif"
    lat=read_tiff(in_file_lat)

    latrad=lat*radperdegree
    sloperad=slope*radperdegree
    aspectrad=aspect*radperdegree
    h=(time-12.0)*15.0*!PI/180.0

    dist=1.0+0.033*cos(doys[i]*2*!pi/365.0)
    decl = -0.4092797 * cos(((doys[i]-1.0) + 11.25) * 0.017214)
    tau=0.68+2*0.00001*dem 


    bsg1 = -sin(sloperad) * sin(aspectrad) * cos(decl)
    bsg2 = (-cos(aspectrad) * sin(sloperad) * sin(latrad) + cos(sloperad) * cos(latrad)) * cos(decl)
    bsg3 = (cos(aspectrad) * sin(sloperad) * cos(latrad) + cos(sloperad) * sin(latrad)) * sin(decl)
    cosza = sin(h) * bsg1 + cos(h) * bsg2 + bsg3

    global=1367.0*tau*cosza*dist

    h_rise=-acos( -tan(decl)*tan(latrad))
    Dtime=(h-h_rise)/(radperdegree*15.0)
    N=(-h_rise)*2.0/(radperdegree*15.0)

    in_file = dirin_Sdn[i]

    sdn= read_tiff(in_file, geotiff = GeoKeys)
    sdn_1=sdn*86400
    global_Jday=global*2*N*3600.0/(!pi*sin(!pi*Dtime/N))
    realtime=N
    sigma_d=4.89e-3     
    Net_Jlongwave=sigma_d*Tak^4*(0.56-0.08*sqrt(es/100))*(0.1+0.9*realtime/N)*0.6 
    Rn_Jday=sdn_1-Net_Jlongwave


    global_day=global_Jday/(N*3600)
    Net_longwave=Net_Jlongwave/(N*3600)
    Rn_day=Rn_Jday/(N*3600)


    out=where(Rn_day lt 0.0, count)
    if count ne 0 then Rn_day(out)=0.0
    Rn_c_day = fc*Rn_day
    Rn_s_day = (1-fc)*Rn_day
 
    LE_c_Daily = 1.2*EF_c*Rn_c_day*N*3600/(2.43*1000000)*(landcover_data ge 11 or landcover_data le 8) + 0.7*EF_c*Rn_c_day*N*3600/(2.43*1000000)*(landcover_data ge 9 and landcover_data le 10)

    LE_s_Daily = 0.8*EF_s*Rn_s_day*N*3600/(2.43*1000000)
    LE_c_Daily[where(~finite(LE_c_Daily))] = 0.0
    LE_s_Daily[where(~finite(LE_s_Daily))] = 0.0

    LE_c_Daily= (LE_c_Daily lt 7.0)*LE_c_Daily+(EF_c ge 7.0)*7.0
    LE_s_Daily= (LE_s_Daily lt 4.0)*LE_s_Daily+(LE_s_Daily ge 4.0)*4.0

    LE_Daily =  LE_c_Daily + LE_s_Daily


    write_tiff, dirout2 + year + doys[i] + '_daily_LEs.tif', LE_s_Daily, GEOTIFF = GeoKeys, /float
    write_tiff, dirout3 + year + doys[i] + '_daily_LEc.tif', LE_c_Daily, GEOTIFF = GeoKeys, /float
    write_tiff, dirout_LE + year + doys[i] + '_daily_LE1.tif', LE_Daily, GEOTIFF = GeoKeys, /float
    write_tiff, dirout_Rn + year + doys[i] + '_daily_Rn.tif', Rn_day, GEOTIFF = GeoKeys, /float
    
    
    print, doys[i]
  endfor

  print, 'RnG complete'

end
