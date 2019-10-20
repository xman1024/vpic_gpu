#define IN_rng
#include "frandn_table.h"

const float frandn_zig_x[65] = {
    0.0000000000000000000000000000000000000000e+00f,   // 0
    3.4555202242398008749546534090768545866013e-01f,   // 1
    4.6215568102338339118162480190221685916185e-01f,   // 2
    5.4504600696136429327509631548309698700905e-01f,   // 3
    6.1195362523637453211478032244485802948475e-01f,   // 4
    6.6931788089916532946688221272779628634453e-01f,   // 5
    7.2027435992699895273005950002698227763176e-01f,   // 6
    7.6660874190091954361037096532527357339859e-01f,   // 7
    8.0944453838211105622946206494816578924656e-01f,   // 8
    8.4953965796918218256195132198627106845379e-01f,   // 9
    8.8743255561541500853195429954212158918381e-01f,   // 10
    9.2352148166420278130317456088960170745850e-01f,   // 11
    9.5811067502319913202768475457560271024704e-01f,   // 12
    9.9143886110667567290732904439209960401058e-01f,   // 13
    1.0236976588316564207303827060968615114689e+00f,   // 14
    1.0550439302185798950972639431711286306381e+00f,   // 15
    1.0856083361085464566997416113736107945442e+00f,   // 16
    1.1155014292694087618684761764598079025745e+00f,   // 17
    1.1448180996627574934620952262775972485542e+00f,   // 18
    1.1736408879046436037896228299359790980816e+00f,   // 19
    1.2020425036545492325501527375308796763420e+00f,   // 20
    1.2300877745426082032764725227025337517262e+00f,   // 21
    1.2578351804105705191716424451442435383797e+00f,   // 22
    1.2853380813622194978051993530243635177612e+00f,   // 23
    1.3126457172210777013532378987292759120464e+00f,   // 24
    1.3398040349740047982862733988440595567226e+00f,   // 25
    1.3668563862516427409588004593388177454472e+00f,   // 26
    1.3938441267285319735691473397309891879559e+00f,   // 27
    1.4208071421486114793708566139684990048409e+00f,   // 28
    1.4477843206031748568562989021302200853825e+00f,   // 29
    1.4748139871194314576285933071631006896496e+00f,   // 30
    1.5019343141686898324138610405498184263706e+00f,   // 31
    1.5291837201179598881850552061223424971104e+00f,   // 32
    1.5566012667655266810129432997200638055801e+00f,   // 33
    1.5842270668274167633171600755304098129272e+00f,   // 34
    1.6121027125411591107706499315099790692329e+00f,   // 35
    1.6402717374392337124078267152071930468082e+00f,   // 36
    1.6687801248810121279575469088740646839142e+00f,   // 37
    1.6976768792403378327549035020638257265091e+00f,   // 38
    1.7270146789200011561149494809797033667564e+00f,   // 39
    1.7568506348952772633253971434896811842918e+00f,   // 40
    1.7872471847043158721390909704496152698994e+00f,   // 41
    1.8182731603303214917843888542847707867622e+00f,   // 42
    1.8500050801824032831177646585274487733841e+00f,   // 43
    1.8825287317531524955427357781445607542992e+00f,   // 44
    1.9159411345865378084596386543125845491886e+00f,   // 45
    1.9503530061152496433152236932073719799519e+00f,   // 46
    1.9858919007063238204580102319596335291862e+00f,   // 47
    2.0227062628527665566480209236033260822296e+00f,   // 48
    2.0609707418998253203312742698471993207932e+00f,   // 49
    2.1008932798907395955723131919512525200844e+00f,   // 50
    2.1427247439335594947351637529209256172180e+00f,   // 51
    2.1867722976565664438908243027981370687485e+00f,   // 52
    2.2334184185755194818057134398259222507477e+00f,   // 53
    2.2831487132108820858888975635636597871780e+00f,   // 54
    2.3365939557574555429653173632686957716942e+00f,   // 55
    2.3945961497601180312244650849606841802597e+00f,   // 56
    2.4583173610577215839612108538858592510223e+00f,   // 57
    2.5294298159785535418109247984830290079117e+00f,   // 58
    2.6104736490220101785553197260014712810516e+00f,   // 59
    2.7055999860695298941948294668691232800484e+00f,   // 60
    2.8223424973239761293086758087156340479851e+00f,   // 61
    2.9768237987905612484951234364416450262070e+00f,   // 62
    3.2159292455085228233657712593185351579450e+00f,   // 63
    3.5268813603277883051562158200908925209660e+00f};  // 64

const float frandn_zig_y[65] = {
    1.0000000000000000000000000000000000000000e+00f,   // 0
    9.4204418489164818507680329573439337309537e-01f,   // 1
    8.9871084518764474558666366377224221650977e-01f,   // 2
    8.6196761825608605203348847045852210158046e-01f,   // 3
    8.2924169214659409110797227970834910593112e-01f,   // 4
    7.9932055946294713378880852916452681711235e-01f,   // 5
    7.7151622512019601854513922711831241940672e-01f,   // 6
    7.4539240467919501453072234076202562391700e-01f,   // 7
    7.2065105654192787809587603109484632568638e-01f,   // 8
    6.9707740823244799719358988365769391748472e-01f,   // 9
    6.7451034215492141851190643908608990386711e-01f,   // 10
    6.5282514097710719761305520059124773979420e-01f,   // 11
    6.3192280720294644763036784329557349337847e-01f,   // 12
    6.1172312580295890266195157902728851695429e-01f,   // 13
    5.9215997749530544576757745689477019368496e-01f,   // 14
    5.7317806731288086608900114726772301310120e-01f,   // 15
    5.5473057713082416246477140964188379257394e-01f,   // 16
    5.3677744090307613671281608547225516758772e-01f,   // 17
    5.1928405123023571848123181027290229394566e-01f,   // 18
    5.0222027190189616680567746831442832444736e-01f,   // 19
    4.8555967208031390367037501143432365324770e-01f,   // 20
    4.6927892404233917991165578320611473372992e-01f,   // 21
    4.5335732363410094483886260408045387748643e-01f,   // 22
    4.3777640417616599022082243819475877444347e-01f,   // 23
    4.2251962250295206272423034765672866797104e-01f,   // 24
    4.0757210137363272404922159841689222048444e-01f,   // 25
    3.9292041643924028389463221078159449461964e-01f,   // 26
    3.7855241880027493887699019392556465390953e-01f,   // 27
    3.6445708627566789607134419004541570075162e-01f,   // 28
    3.5062439805206711601528318800191641457786e-01f,   // 29
    3.3704522854538442405377719757186838478447e-01f,   // 30
    3.2371125719058124381899595178868622724622e-01f,   // 31
    3.1061489155547205394045096049637777468888e-01f,   // 32
    2.9774920170315998462591197681881283187977e-01f,   // 33
    2.8510786414412745597493636928732030355604e-01f,   // 34
    2.7268511405126700784828409906968005316230e-01f,   // 35
    2.6047570468036198649197307530833711552987e-01f,   // 36
    2.4847487316078160514269158876121679213611e-01f,   // 37
    2.3667831200900182462198897276328679595281e-01f,   // 38
    2.2508214588119269128634221399654080641994e-01f,   // 39
    2.1368291322922901786206251184552229460678e-01f,   // 40
    2.0247755266505245362021383342954194972663e-01f,   // 41
    1.9146339397930083145882990125485889620904e-01f,   // 42
    1.8063815391025909397792938854143685034614e-01f,   // 43
    1.6999993692895921206843529854824126346102e-01f,   // 44
    1.5954724150925197811512826207014370538673e-01f,   // 45
    1.4927897260658239165251413155255377773756e-01f,   // 46
    1.3919446140292705076080791160020311281187e-01f,   // 47
    1.2929349382809189428309316199916523260072e-01f,   // 48
    1.1957635000126569613200164243771439487318e-01f,   // 49
    1.1004385764974420347799173580671805439124e-01f,   // 50
    1.0069746391502830031711417796413954306445e-01f,   // 51
    9.1539332022017442505243400852066315565025e-02f,   // 52
    8.2572472540893222921144691678341231977356e-02f,   // 53
    7.3800924281228182179971902127091354373079e-02f,   // 54
    6.5230008879955892272953579813510316398606e-02f,   // 55
    5.6866699215431624441342926129516754940596e-02f,   // 56
    4.8720172066754372234615886802511219855205e-02f,   // 57
    4.0802676591920000420638881336032888214049e-02f,   // 58
    3.3130984855278960237964010898670252913689e-02f,   // 59
    2.5729022545612278111652878806459021632236e-02f,   // 60
    1.8633232739481427416337140298696617435326e-02f,   // 61
    1.1905676634192996061346718223578067252788e-02f,   // 62
    5.6783166417766997683490141392292915867301e-03f,   // 63
    1.9903474352364905982225876330052316554031e-03f};  // 64
